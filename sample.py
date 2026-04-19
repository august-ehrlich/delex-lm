#!/usr/bin/env python3
"""
sample.py — Generate surface-form text from a trained DelexLM / FactoredDelexLM.

For each model token sampled autoregressively:
  Structural token  [en|VERB|frame3|Mood=Ind|Tense=Past]
      → pick a surface form from surface_dict.json weighted by corpus counts
  Function word     en::the
      → strip the lang:: prefix and emit the surface form directly
  End of sentence   <eos>
      → emit a newline and optionally start a fresh context

Spacing heuristic: no space before common punctuation (.,!?;:)'" etc.).
Works for qualitative evaluation ("is the output grammatically plausible?")
even though it won't be semantically coherent.

Usage:
    # Standard mode
    python sample.py --checkpoint runs/run1/best_model.pt \\
        --vocab runs/run1/vocab.json \\
        --surface-dict surface_dict.json \\
        --n-sentences 10

    # Factored mode
    python sample.py --checkpoint runs/run1/best_model.pt \\
        --factor-vocab factor_vocab.json \\
        --surface-dict surface_dict.json \\
        --n-sentences 10 --temperature 0.8 --top-k 50
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Spacing heuristic
# ---------------------------------------------------------------------------

# Tokens that should NOT be preceded by a space.
NO_SPACE_BEFORE = set(".,!?;:)'\"»›—–")
NO_SPACE_BEFORE.update(["...", "n't", "'s", "'re", "'ve", "'ll", "'d", "'m"])

# Tokens after which the next token needs no leading space.
NO_SPACE_AFTER = set("(\"'«‹")


def tokens_to_text(surface_tokens: list[str]) -> str:
    """Join surface tokens with spaces, suppressing space around punctuation."""
    parts: list[str] = []
    suppress_next = False
    for i, tok in enumerate(surface_tokens):
        if suppress_next or tok in NO_SPACE_BEFORE:
            parts.append(tok)
        else:
            parts.append((" " if parts else "") + tok)
        suppress_next = tok in NO_SPACE_AFTER
    return "".join(parts)


# ---------------------------------------------------------------------------
# Surface form sampling
# ---------------------------------------------------------------------------

def sample_surface(
    delex_token: str,
    surface_dict: dict[str, dict[str, int]],
    rng: random.Random,
) -> str:
    """
    Return a surface form for delex_token, sampled from the corpus distribution.
    Falls back to the token itself (stripped of brackets) if unknown.
    """
    forms = surface_dict.get(delex_token)
    if not forms:
        # Fallback: strip brackets and lang prefix for a readable placeholder
        inner = delex_token.strip("[]")
        parts = inner.split("|")
        # Remove lang code (all-lowercase, no '=') if present
        if parts and "=" not in parts[0] and parts[0].islower() and parts[0].isalpha():
            parts = parts[1:]
        return f"<{parts[0] if parts else '?'}>"

    population = list(forms.keys())
    weights    = list(forms.values())
    return rng.choices(population, weights=weights, k=1)[0]


def delex_to_surface(
    token: str,
    surface_dict: dict[str, dict[str, int]],
    rng: random.Random,
) -> str | None:
    """
    Convert a single delex token to its surface form.
    Returns None for <eos> (sentence boundary signal).
    """
    if token == "<eos>":
        return None
    if token.startswith("[") and token.endswith("]"):
        return sample_surface(token, surface_dict, rng)
    if "::" in token:
        return token.split("::", 1)[1]   # strip lang:: prefix
    return token   # bare surface (legacy, no lang prefix)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device, factor_vocab_path: Path | None = None):
    """Load checkpoint and return (model, is_factored)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from model import DelexLM, FactoredDelexLM

    factored    = ckpt.get("factored", False)
    d_model     = ckpt["d_model"]
    n_layers    = ckpt["n_layers"]
    n_heads     = ckpt["n_heads"]
    d_ff        = ckpt["d_ff"]
    context_len = ckpt["context_len"]

    if factored:
        from build_factor_vocab import FactorVocab
        # Prefer --factor-vocab CLI arg; fall back to path stored in checkpoint.
        resolved = factor_vocab_path or Path(ckpt.get("factor_vocab", ""))
        if not resolved or not resolved.exists():
            raise SystemExit(
                f"Cannot find factor_vocab.json (tried {resolved}). "
                "Pass --factor-vocab <path>."
            )
        fv = FactorVocab.load(resolved)
        model = FactoredDelexLM(
            factor_vocab=fv,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            context_len=context_len,
            dropout=0.0,
        )
    else:
        vocab_size = ckpt["vocab_size"]
        model = DelexLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            context_len=context_len,
            dropout=0.0,
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, factored


def load_vocab_mappings(
    ckpt_path: Path,
    vocab_path: Path | None,
    factor_vocab_path: Path | None,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Return (tok_to_id, id_to_tok).
    Mode is determined by whether factor_vocab_path was supplied:
      factored  → load from factor_vocab.json
      standard  → load from vocab.json (sibling of checkpoint if not specified)
    """
    if factor_vocab_path is not None:
        with open(factor_vocab_path, encoding="utf-8") as f:
            fv = json.load(f)
        tok_to_id = fv["token_to_id"]
        id_to_tok = {int(k): v for k, v in fv["id_to_token"].items()}
    else:
        if vocab_path is None:
            vocab_path = ckpt_path.parent / "vocab.json"
        if not vocab_path.exists():
            raise SystemExit(f"vocab.json not found at {vocab_path}; pass --vocab")
        with open(vocab_path, encoding="utf-8") as f:
            idx2tok = json.load(f)   # list[str]
        tok_to_id = {tok: i for i, tok in enumerate(idx2tok)}
        id_to_tok = {i: tok for i, tok in enumerate(idx2tok)}

    return tok_to_id, id_to_tok


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_sentences(
    model,
    tok_to_id: dict[str, int],
    id_to_tok: dict[int, str],
    surface_dict: dict[str, dict[str, int]],
    n_sentences: int,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    device: torch.device,
    lang_filter: str | None,
) -> list[str]:
    """
    Generate n_sentences by repeated autoregressive sampling.

    Each generation starts from a single <eos> token (sentence boundary).
    Tokens are collected until the next <eos> or max_tokens is reached.
    """
    rng = random.Random(seed)
    eos_id = tok_to_id.get("<eos>")
    if eos_id is None:
        raise SystemExit("<eos> not found in vocabulary")

    sentences: list[str] = []

    for _ in range(n_sentences):
        ctx = torch.tensor([[eos_id]], dtype=torch.long, device=device)
        surface_toks: list[str] = []
        delex_toks: list[str] = []

        for _ in range(max_tokens):
            with torch.no_grad():
                logits, _ = model(ctx[:, -model.context_len:])
            logits = logits[0, -1, :] / temperature   # (V,)

            if lang_filter:
                # Zero out logits for tokens from other languages.
                # Structural: [lang|...], function: lang::
                for tok_id, tok in id_to_tok.items():
                    if tok == "<eos>":
                        continue
                    tok_lang = None
                    if tok.startswith("["):
                        inner = tok[1:].split("|")[0]
                        if inner.isalpha() and inner.islower():
                            tok_lang = inner
                    elif "::" in tok:
                        tok_lang = tok.split("::")[0]
                    if tok_lang and tok_lang != lang_filter:
                        logits[tok_id] = float("-inf")

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[-1], float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            delex_tok = id_to_tok[next_id]

            if delex_tok == "<eos>":
                break

            surf = delex_to_surface(delex_tok, surface_dict, rng)
            if surf is not None:
                surface_toks.append(surf)
                delex_toks.append(delex_tok)

            ctx = torch.cat([ctx, torch.tensor([[next_id]], device=device)], dim=1)

        sentences.append((surface_toks, delex_toks))

    return sentences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sample surface-form text from a trained DelexLM for qualitative eval.",
    )
    parser.add_argument("--checkpoint",    required=True, metavar="FILE",
                        help="Path to best_model.pt")
    parser.add_argument("--vocab",         default=None, metavar="FILE",
                        help="[standard mode] vocab.json (default: sibling of checkpoint)")
    parser.add_argument("--factor-vocab",  default=None, metavar="FILE",
                        help="[factored mode] factor_vocab.json")
    parser.add_argument("--surface-dict",  default="surface_dict.json", metavar="FILE",
                        help="surface_dict.json from build_surface_dict.py")
    parser.add_argument("--n-sentences",   type=int, default=20)
    parser.add_argument("--max-tokens",    type=int, default=50,
                        help="Max tokens per sentence before cutting off")
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--top-k",         type=int, default=None,
                        help="Top-k truncation (e.g. 50). None = unrestricted.")
    parser.add_argument("--lang",          default=None, metavar="ISO",
                        help="Only sample tokens for this language (e.g. 'en'). "
                             "Useful for cleaner qualitative eval.")
    parser.add_argument("--show-delex",    action="store_true",
                        help="Print the delex token sequence alongside the surface text")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--device",        default="cpu",
                        help="torch device (cpu / cuda / cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, factored = load_model(
        Path(args.checkpoint),
        device,
        factor_vocab_path=Path(args.factor_vocab) if args.factor_vocab else None,
    )
    d = model.pos_emb.embedding_dim
    print(f"  Mode: {'factored' if factored else 'standard'}  |  "
          f"d_model={d}  |  params={model.num_params():,}")

    tok_to_id, id_to_tok = load_vocab_mappings(
        Path(args.checkpoint),
        Path(args.vocab) if args.vocab else None,
        Path(args.factor_vocab) if args.factor_vocab else None,
    )
    print(f"  Vocab: {len(tok_to_id):,} tokens")

    print(f"Loading surface dict: {args.surface_dict}")
    with open(args.surface_dict, encoding="utf-8") as f:
        surface_dict = json.load(f)
    print(f"  {len(surface_dict):,} token types with surface forms")

    if args.lang:
        print(f"  Language filter: {args.lang}")

    print(f"\nGenerating {args.n_sentences} sentences "
          f"(temp={args.temperature}, top_k={args.top_k})...\n")
    print("=" * 72)

    results = generate_sentences(
        model       = model,
        tok_to_id   = tok_to_id,
        id_to_tok   = id_to_tok,
        surface_dict= surface_dict,
        n_sentences = args.n_sentences,
        max_tokens  = args.max_tokens,
        temperature = args.temperature,
        top_k       = args.top_k,
        seed        = args.seed,
        device      = device,
        lang_filter = args.lang,
    )

    for i, (surface_toks, delex_toks) in enumerate(results, 1):
        text = tokens_to_text(surface_toks)
        print(f"[{i:02d}] {text}")
        if args.show_delex:
            print(f"      {' '.join(delex_toks)}")
        print()

    print("=" * 72)


if __name__ == "__main__":
    main()
