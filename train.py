#!/usr/bin/env python3
"""
train.py — Training script for DelexLM and FactoredDelexLM.

Trains until dev perplexity stops improving, logging train and dev
perplexity each epoch so you can track the memorisation/generalisation
timeline. The best checkpoint (by dev PPL) is saved to disk.

Modes:
  Standard  (default):  flat token vocab, learned embedding table.
  Factored  (--factored --factor-vocab path/factor_vocab.json):
                         multi-hot feature embedding (FactoredDelexLM).

Usage:
    python train.py --data-dir output/ --out-dir runs/run1/
    python train.py --data-dir output/ --out-dir runs/run1/ \\
        --factored --factor-vocab factor_vocab.json
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model import DelexLM


# ---------------------------------------------------------------------------
# Vocabulary (standard mode)
# ---------------------------------------------------------------------------

class Vocab:
    """
    Simple whitespace-token vocabulary built from a flat token list.

    Reserved tokens:
        <unk>  unknown tokens seen at inference but not at training time
    """
    UNK = "<unk>"

    def __init__(self, token_counts: Counter, min_freq: int = 1):
        # Sort by descending frequency then alphabetically for determinism
        ordered = [tok for tok, cnt in token_counts.most_common() if cnt >= min_freq]
        self.idx2tok: list[str] = [self.UNK] + ordered
        self.tok2idx: dict[str, int] = {tok: i for i, tok in enumerate(self.idx2tok)}

    def __len__(self) -> int:
        return len(self.idx2tok)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.tok2idx[self.UNK]
        return [self.tok2idx.get(t, unk) for t in tokens]

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.idx2tok, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        obj = cls.__new__(cls)
        with open(path, encoding="utf-8") as f:
            obj.idx2tok = json.load(f)
        obj.tok2idx = {tok: i for i, tok in enumerate(obj.idx2tok)}
        return obj


def read_tokens(path: Path) -> list[str]:
    """Read a delexicalized .txt file and return a flat list of tokens."""
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens.extend(line.split())
    return tokens


def load_lang_dev_loaders(
    data_dir: Path,
    encode,
    context_len: int,
    batch_size: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    """
    Build one DataLoader per language for per-language dev PPL.

    Tries two data layouts in order:
      1. langs/{iso}/dev.txt  — ISO-keyed layout produced by delexicalize.py
                                and stored in the HuggingFace repo
      2. UD_{Language}-{Corpus}/dev.txt — legacy treebank-dir layout

    In layout 2, dev tokens are concatenated across all treebanks of the
    same language, and the key is the UD language name (e.g. "English").
    In layout 1, the key is the ISO code (e.g. "en").

    `encode` is a callable list[str] → list[int] (works for both standard
    Vocab.encode and FactorVocab token_to_id lookup).

    Languages with fewer than context_len + 1 tokens are silently skipped.
    """
    lang_tokens: dict[str, list[str]] = {}

    langs_dir = data_dir / "langs"
    if langs_dir.is_dir():
        # Layout 1: langs/{iso}/dev.txt
        for lang_dir in sorted(langs_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            dev_file = lang_dir / "dev.txt"
            if dev_file.exists():
                lang_tokens.setdefault(lang_dir.name, []).extend(read_tokens(dev_file))
    else:
        # Layout 2: UD_{Language}-{Corpus}/dev.txt (legacy)
        EXCLUDE_LANGS = {"Classical_Chinese", "Ancient_Greek", "Latin"}
        EXCLUDE_DIRS  = {"UD_Italian-Old", "UD_Swedish-Old"}
        for d in sorted(data_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("UD_"):
                continue
            if d.name in EXCLUDE_DIRS:
                continue
            dev_file = d / "dev.txt"
            if not dev_file.exists():
                continue
            lang = d.name[3:].split("-")[0]
            if lang in EXCLUDE_LANGS:
                continue
            lang_tokens.setdefault(lang, []).extend(read_tokens(dev_file))

    loaders: dict[str, DataLoader] = {}
    for lang, tokens in sorted(lang_tokens.items()):
        ids = encode(tokens)
        if len(ids) < context_len + 1:
            continue
        ds = LMDataset(ids, context_len)
        loaders[lang] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory,
        )
    return loaders


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LMDataset(Dataset):
    """
    Non-overlapping block dataset for next-token prediction.

    All tokens are concatenated into one long sequence (sentence boundaries
    are marked by <eos> tokens already present in the data). The sequence is
    divided into non-overlapping blocks of context_len; each block is one
    training example.

    Blocks are pre-materialised as a (n_chunks, context_len) tensor so that
    DataLoader indexing is a fast 2D row lookup.
    """

    def __init__(self, token_ids: list[int], context_len: int):
        data = torch.tensor(token_ids, dtype=torch.long)
        n_chunks = (len(data) - 1) // context_len
        self.x = data[: n_chunks * context_len].view(n_chunks, context_len)
        self.y = data[1 : n_chunks * context_len + 1].view(n_chunks, context_len)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    """Return mean per-token cross-entropy loss over the entire loader."""
    model.eval()
    total_loss = total_tokens = 0
    amp_enabled = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def make_lr_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_frac: float = 0.1,
):
    """
    Linear warmup then cosine decay.

    Returns a function f(step) -> multiplier for use with LambdaLR.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine
    return lr_lambda


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data_dir = Path(args.data_dir)
    train_path = data_dir / "all_train.txt"
    dev_path   = data_dir / "all_dev.txt"

    # ---- Vocabulary / token encoding ----------------------------------------
    if args.factored:
        from build_factor_vocab import FactorVocab
        from model import FactoredDelexLM
        if not args.factor_vocab:
            raise SystemExit("--factor-vocab is required with --factored")
        fv = FactorVocab.load(Path(args.factor_vocab))
        print(f"FactorVocab loaded: {len(fv.token_to_id):,} tokens, "
              f"{len(fv.feature_to_id):,} features")
        # Encode using flat token vocab; same integer IDs as standard mode
        def encode(tokens: list[str]) -> list[int]:
            unk_id = fv.token_to_id.get("<unk>", 0)
            return [fv.token_to_id.get(t, unk_id) for t in tokens]
        vocab_len = len(fv.token_to_id)
    else:
        print("Reading training data and building vocabulary...")
        train_tokens_for_vocab = read_tokens(train_path)
        vocab = Vocab(Counter(train_tokens_for_vocab), min_freq=args.min_freq)
        print(f"Vocabulary size: {len(vocab):,}")
        encode = vocab.encode
        vocab_len = len(vocab)

    # ---- Encode and build datasets ------------------------------------------
    print("Encoding training data...")
    train_ids = encode(read_tokens(train_path))
    dev_ids   = encode(read_tokens(dev_path))

    train_ds = LMDataset(train_ids, args.context_len)
    dev_ds   = LMDataset(dev_ids,   args.context_len)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin)

    print(f"Train : {len(train_ids):>10,} tokens  ->  {len(train_ds):>8,} chunks")
    print(f"Dev   : {len(dev_ids):>10,} tokens  ->  {len(dev_ds):>8,} chunks")

    # Per-language dev loaders (both standard and factored modes)
    lang_loaders = load_lang_dev_loaders(
        data_dir, encode, args.context_len, args.batch_size, pin_memory=pin,
    )
    if lang_loaders:
        print(f"Lang  : {len(lang_loaders)} languages with dev data")
    print()

    # ---- Model --------------------------------------------------------------
    if args.factored:
        model = FactoredDelexLM(
            factor_vocab = fv,
            d_model      = args.d_model,
            n_layers     = args.n_layers,
            n_heads      = args.n_heads,
            d_ff         = args.d_ff,
            context_len  = args.context_len,
            dropout      = args.dropout,
        ).to(device)
    else:
        model = DelexLM(
            vocab_size  = vocab_len,
            d_model     = args.d_model,
            n_layers    = args.n_layers,
            n_heads     = args.n_heads,
            d_ff        = args.d_ff,
            context_len = args.context_len,
            dropout     = args.dropout,
        ).to(device)
    print(model.summary(), "\n")

    # ---- Optimiser + LR schedule (linear warmup + cosine decay) -------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch  = math.ceil(len(train_ds) / (args.batch_size * args.accum_steps))
    total_steps      = steps_per_epoch * args.max_epochs
    warmup_steps     = steps_per_epoch * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(warmup_steps, total_steps),
    )

    # ---- Output dir ---------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.factored:
        vocab.save(out_dir / "vocab.json")
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- AMP scaler (no-op on CPU) ------------------------------------------
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- Training loop (with gradient accumulation) -------------------------
    best_dev_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    log: list[dict] = []

    col = f"{'Epoch':>6}  {'Train PPL':>10}  {'Dev PPL':>10}  {'LR':>9}  {'Time':>7}"
    print(col)
    print("-" * len(col))

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()

        # --- Train one epoch -------------------------------------------------
        model.train()
        total_loss = total_tokens = 0
        optimizer.zero_grad()
        step_in_epoch = 0

        for batch_i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                _, loss = model(x, y)
            loss = loss / args.accum_steps
            scaler.scale(loss).backward()

            if (batch_i + 1) % args.accum_steps == 0 or (batch_i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_in_epoch += 1

            n = y.numel()
            total_loss += loss.item() * args.accum_steps * n
            total_tokens += n

        # --- Evaluate --------------------------------------------------------
        train_loss = total_loss / total_tokens
        dev_loss   = evaluate(model, dev_loader, device)

        train_ppl = math.exp(train_loss)
        dev_ppl   = math.exp(dev_loss)
        lr_now    = scheduler.get_last_lr()[0]
        elapsed   = time.time() - t0

        print(f"{epoch:>6}  {train_ppl:>10.2f}  {dev_ppl:>10.2f}  {lr_now:>9.2e}  {elapsed:>5.1f}s")

        lang_ppls: dict[str, float] = {}
        if lang_loaders:
            for lang, loader in lang_loaders.items():
                lang_ppls[lang] = math.exp(evaluate(model, loader, device))
            langs_sorted = sorted(lang_ppls.items())
            for i in range(0, len(langs_sorted), 3):
                row = langs_sorted[i : i + 3]
                print("  " + "   ".join(f"{l:<22s}{p:>7.2f}" for l, p in row))

        log.append({
            "epoch":     epoch,
            "train_ppl": round(train_ppl, 4),
            "dev_ppl":   round(dev_ppl,   4),
            "lr":        round(lr_now, 8),
            "elapsed_s": round(elapsed, 2),
            "lang_ppl":  {l: round(p, 4) for l, p in lang_ppls.items()},
        })

        # --- Checkpoint + early stopping -------------------------------------
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch    = epoch
            patience_ctr  = 0
            ckpt = {
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "d_model":      args.d_model,
                "n_layers":     args.n_layers,
                "n_heads":      args.n_heads,
                "d_ff":         args.d_ff,
                "context_len":  args.context_len,
                "factored":     args.factored,
            }
            if args.factored:
                ckpt["factor_vocab"] = str(args.factor_vocab)
                ckpt["n_features"]   = len(fv.feature_to_id)
                ckpt["n_tokens"]     = len(fv.token_to_id)
            else:
                ckpt["vocab_size"] = vocab_len
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping: dev PPL did not improve for "
                      f"{args.patience} epochs.")
                break

    # ---- Persist training log -----------------------------------------------
    log_path = out_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nBest dev PPL : {math.exp(best_dev_loss):.2f}  (epoch {best_epoch})")
    print(f"Checkpoint   : {out_dir / 'best_model.pt'}")
    print(f"Training log : {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DelexLM (or FactoredDelexLM) on delexicalized UD data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--data-dir",     required=True, metavar="DIR")
    parser.add_argument("--out-dir",      required=True, metavar="DIR")

    # Factored input mode
    parser.add_argument("--factored",     action="store_true",
                        help="Use FactoredDelexLM with multi-hot feature embeddings")
    parser.add_argument("--factor-vocab", metavar="FILE", default=None,
                        help="Path to factor_vocab.json (required with --factored)")

    # Model architecture
    parser.add_argument("--d-model",     type=int,   default=256)
    parser.add_argument("--n-layers",    type=int,   default=4)
    parser.add_argument("--n-heads",     type=int,   default=8)
    parser.add_argument("--d-ff",        type=int,   default=512)
    parser.add_argument("--context-len", type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.1)

    # Training
    parser.add_argument("--batch-size",    type=int,   default=128,
                        help="Per-device batch size (before gradient accumulation)")
    parser.add_argument("--accum-steps",   type=int,   default=1,
                        help="Gradient accumulation steps (effective batch = batch_size x accum_steps)")
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int,   default=2,
                        help="Epochs of linear LR warmup before cosine decay begins")
    parser.add_argument("--weight-decay",  type=float, default=0.1)
    parser.add_argument("--grad-clip",     type=float, default=1.0)
    parser.add_argument("--max-epochs",    type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=8,
                        help="Early stopping patience in epochs")
    parser.add_argument("--min-freq",      type=int,   default=3,
                        help="[standard mode] Min token frequency to include in vocab")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
