"""
model.py — Transformer language model for delexicalized UD sequences.

Architecture per block:
    Pre-LN  →  causal multi-head self-attention (8 heads)  →  residual
    Pre-LN  →  3-layer FFN  (d_model → 512 → 512 → d_model, GELU)  →  residual

Stack:        4 such blocks
Context:      128 tokens
Input:        learned token embeddings + learned positional embeddings
              (embedding table is tiny given the small vocabulary)
Output:       linear head weight-tied to the token embedding matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a fixed lower-triangular causal mask."""

    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Causal mask: True where attention is allowed (lower-triangular)
        mask = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)  # (L, L), not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V and split into heads
        q, k, v = self.qkv(x).split(C, dim=-1)                          # each (B, T, C)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)     # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal masking
        att = (q @ k.transpose(-2, -1)) * self.scale                    # (B, H, T, T)
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum over values, merge heads
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C)
        return self.out_proj(y)


class FFN(nn.Module):
    """
    3-layer feed-forward network with GELU activations.

    Shape:  d_model  →  d_ff  →  d_ff  →  d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff,    d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block.

        x  →  x + Attn(LN(x))  →  x + FFN(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, context_len: int, dropout: float):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn    = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln_ffn  = nn.LayerNorm(d_model)
        self.ffn     = FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DelexLM(nn.Module):
    """
    Autoregressive transformer language model for delexicalized UD sequences.

    Args:
        vocab_size:  number of tokens in the vocabulary
        d_model:     embedding / hidden dimension            (default 256)
        n_layers:    number of transformer blocks            (default 4)
        n_heads:     attention heads per block               (default 8)
        d_ff:        FFN hidden dimension                    (default 512)
        context_len: maximum sequence length                 (default 128)
        dropout:     dropout rate applied throughout         (default 0.0)
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 256,
        n_layers:    int   = 4,
        n_heads:     int   = 8,
        d_ff:        int   = 512,
        context_len: int   = 128,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.context_len = context_len
        self.vocab_size  = vocab_size

        # ---- Input representations ----------------------------------------
        # Token embedding table is tiny (vocab_size * d_model) given the small
        # vocabulary of this project, but we keep a learned table for efficient
        # lookup and weight tying with the output head.
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(context_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # ---- Transformer body ---------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, context_len, dropout)
            for _ in range(n_layers)
        ])

        # ---- Output -------------------------------------------------------
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: the output projection reuses the token embedding matrix,
        # so predicting a token and embedding it share the same representation.
        self.head.weight = self.token_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx:     torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:     (B, T) token indices,  T ≤ context_len
            targets: (B, T) next-token indices for teacher-forcing, or None

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.context_len, (
            f"Input length {T} exceeds model context length {self.context_len}"
        )

        positions = torch.arange(T, device=idx.device)                  # (T,)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.head(x)                                            # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation by sampling.

        Args:
            idx:            (B, T) prompt token indices
            max_new_tokens: number of tokens to append
            temperature:    scales logits before softmax (<1 = sharper)
            top_k:          if set, zero out all but the top-k logits before sampling

        Returns:
            (B, T + max_new_tokens) token indices
        """
        for _ in range(max_new_tokens):
            # Crop context to the model's window
            idx_cond = idx[:, -self.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                     # (B, V)

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[:, [-1]], float("-inf"))

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)        # (B, 1)
            idx        = torch.cat([idx, next_token], dim=1)

        return idx

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """
        Total unique parameter count.
        Weight tying means token_emb and head share one tensor; this counts it once.
        """
        seen  = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def summary(self) -> str:
        d  = self.token_emb.embedding_dim
        H  = next(b.attn.n_heads for b in self.blocks)
        L  = len(self.blocks)
        ff = next(b.ffn.net[0].out_features for b in self.blocks)
        lines = [
            "DelexLM",
            f"  vocab_size  : {self.vocab_size}",
            f"  d_model     : {d}",
            f"  n_layers    : {L}",
            f"  n_heads     : {H}  (d_head = {d // H})",
            f"  d_ff        : {ff}  (3-layer FFN)",
            f"  context_len : {self.context_len}",
            f"  parameters  : {self.num_params():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factored embedding: sum-of-active-feature-embeddings
# ---------------------------------------------------------------------------

class FactoredEmbedding(nn.Module):
    """
    Represent each token as the sum of its active binary feature embeddings.

    Each of the n_features binary features has its own d_model-dimensional
    embedding vector.  Feature IDs are 1-indexed; ID 0 is the padding index
    whose embedding is fixed to zero.

    Input:  (B, T, F) int64 tensor of active feature IDs, zero-padded.
    Output: (B, T, d_model) float32 sum over the F active embeddings.
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        # +1 rows so that index 0 (padding) maps to the zero row.
        self.emb = nn.Embedding(n_features + 1, d_model, padding_idx=0)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        # feature_ids: (B, T, F)
        # emb lookup:  (B, T, F, d_model)
        # sum over F:  (B, T, d_model)
        return self.emb(feature_ids).sum(dim=-2)


# ---------------------------------------------------------------------------
# Factored language model
# ---------------------------------------------------------------------------

class FactoredDelexLM(nn.Module):
    """
    Autoregressive transformer LM with factored multi-hot input embeddings.

    Input representation
    --------------------
    Each token is encoded as a list of active binary feature IDs (1-based,
    0 = padding).  All features are language-namespaced so the model learns
    cross-lingual equivalences from scratch.  The embedding is the sum of
    all active feature vectors — identical in spirit to a bag-of-embeddings
    or a factored LM input layer.

    Output head
    -----------
    Predicts over the flat token vocabulary (not the feature vocabulary).
    No weight tying (factored input space != output token space).

    Args
    ----
    factor_vocab  : FactorVocab instance, used to read n_features / n_tokens
                    and the max_features_per_token dimension.
    d_model       : hidden dimension
    n_layers      : transformer depth
    n_heads       : attention heads
    d_ff          : FFN hidden dimension
    context_len   : maximum sequence length
    dropout       : dropout rate
    max_features  : F dimension of the input tensor; if None, inferred from
                    factor_vocab.max_features_per_token.
    """

    def __init__(
        self,
        factor_vocab: "FactorVocab",  # imported lazily; avoids hard dep for DelexLM users
        d_model:      int   = 256,
        n_layers:     int   = 4,
        n_heads:      int   = 8,
        d_ff:         int   = 512,
        context_len:  int   = 128,
        dropout:      float = 0.0,
        max_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.context_len  = context_len
        self.n_tokens     = len(factor_vocab.token_to_id)
        self.n_features   = len(factor_vocab.feature_to_id)
        self.max_features = max_features or max(
            len(v) for v in factor_vocab.token_features.values()
        )

        # Store feature-ID lists as a registered buffer for fast batch encoding
        # Shape: (n_tokens, max_features) — 0-padded
        feat_matrix = torch.zeros(self.n_tokens, self.max_features, dtype=torch.long)
        for token, tok_id in factor_vocab.token_to_id.items():
            ids = factor_vocab.token_features.get(token, [])
            for col, fid in enumerate(ids[:self.max_features]):
                feat_matrix[tok_id, col] = fid
        self.register_buffer("feat_matrix", feat_matrix)   # (V, F)

        # ---- Input representations ---------------------------------------
        self.feat_emb = FactoredEmbedding(self.n_features, d_model)
        self.pos_emb  = nn.Embedding(context_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # ---- Transformer body --------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, context_len, dropout)
            for _ in range(n_layers)
        ])

        # ---- Output head -------------------------------------------------
        self.ln_f = nn.LayerNorm(d_model)
        # No weight tying: output space (flat tokens) != input space (features)
        self.head = nn.Linear(d_model, self.n_tokens, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx:     torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args
        ----
        idx     : (B, T) token indices (into flat token vocab), T <= context_len
        targets : (B, T) next-token indices for teacher-forcing, or None

        Returns
        -------
        logits : (B, T, n_tokens)
        loss   : scalar cross-entropy if targets given, else None
        """
        B, T = idx.shape
        assert T <= self.context_len

        # Token index -> active feature IDs: (B, T, F)
        feat_ids = self.feat_matrix[idx]                       # (B, T, F)

        positions = torch.arange(T, device=idx.device)        # (T,)
        x = self.emb_drop(
            self.feat_emb(feat_ids) + self.pos_emb(positions)
        )

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.head(x)                                  # (B, T, n_tokens)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.n_tokens),
                targets.view(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[:, [-1]], float("-inf"))

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat([idx, next_token], dim=1)

        return idx

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        seen  = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def summary(self) -> str:
        d  = self.feat_emb.emb.embedding_dim
        H  = next(b.attn.n_heads for b in self.blocks)
        L  = len(self.blocks)
        ff = next(b.ffn.net[0].out_features for b in self.blocks)
        lines = [
            "FactoredDelexLM",
            f"  n_tokens       : {self.n_tokens:,}  (flat output vocab)",
            f"  n_features     : {self.n_features:,}  (binary input features)",
            f"  max_feats/tok  : {self.max_features}",
            f"  d_model        : {d}",
            f"  n_layers       : {L}",
            f"  n_heads        : {H}  (d_head = {d // H})",
            f"  d_ff           : {ff}  (3-layer FFN)",
            f"  context_len    : {self.context_len}",
            f"  parameters     : {self.num_params():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VOCAB = 512   # approximate upper bound; real size determined after delexicalization
    model = DelexLM(vocab_size=VOCAB)
    print(model.summary())

    # Forward pass with a random batch
    B, T = 4, 128
    idx     = torch.randint(0, VOCAB, (B, T))
    targets = torch.randint(0, VOCAB, (B, T))
    logits, loss = model(idx, targets)

    print(f"\nForward pass OK")
    print(f"  logits : {tuple(logits.shape)}  (expected {B}, {T}, {VOCAB})")
    print(f"  loss   : {loss.item():.4f}  (expected ~{torch.log(torch.tensor(VOCAB)).item():.2f} for random init)")
