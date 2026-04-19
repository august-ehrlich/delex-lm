#!/usr/bin/env python3
"""
build_factor_vocab.py — Build per-language binary feature vocabulary from
delexicalized token vocabulary.

Each token in the flat token vocabulary is decomposed into a set of binary
features.  All structural features are language-namespaced so the model must
learn cross-lingual equivalences from scratch.

Feature naming conventions
--------------------------
Token type               Example token          Feature(s)
-----------------------  ---------------------  ----------------------------
Lexeme (with lang)       [en|VERB|frame3|...]   en::UPOS=VERB
                                                en::frame=3
                                                en::Mood=Ind
                                                en::Tense=Past  ...
Lexeme (no lang)         [VERB|Mood=Ind]        UPOS=VERB
                                                Mood=Ind  ...
Function word (namespac) en::the                word=en::the
Function word (bare)     the                    word=the
Special                  <eos>                  special=eos

Output
------
factor_vocab.json with keys:
  feature_to_id   : {feature_str: int}  (1-indexed; 0 reserved for padding)
  id_to_feature   : {int_str: str}
  token_to_id     : {token_str: int}  (0-indexed flat token vocab)
  token_features  : {token_str: [feat_id, ...]}  sparse active features
  n_features      : int  (max feature ID, i.e. len(feature_to_id))
  n_tokens        : int
  max_features_per_token : int

Usage
-----
    python build_factor_vocab.py --vocab output_multilingual/delex_vocab.txt
    python build_factor_vocab.py --vocab output_multilingual/delex_vocab.txt \\
        --output factor_vocab.json
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Token feature parser
# ---------------------------------------------------------------------------

def parse_token_features(token: str) -> list[str]:
    """
    Decompose a delexicalized token into its binary feature names.

    Handles:
      <eos>                         -> ["special=eos"]
      [en|VERB|frame3|Mood=Ind|...] -> ["en::UPOS=VERB", "en::frame=3",
                                         "en::Mood=Ind", ...]
      [VERB|Mood=Ind|...]           -> ["UPOS=VERB", "Mood=Ind", ...]
      en::the                       -> ["word=en::the"]
      the                           -> ["word=the"]
    """
    if token == "<eos>":
        return ["special=eos"]

    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1]
        parts = inner.split("|")

        # Detect language code: first segment is all lowercase alpha, no "="
        lang: str | None = None
        if parts and "=" not in parts[0] and parts[0].isalpha() and parts[0].islower():
            lang = parts[0]
            parts = parts[1:]

        prefix = f"{lang}::" if lang else ""
        features: list[str] = []
        upos_done = False

        for part in parts:
            if "=" in part:
                # Morphological feature: Gender=Masc, Mood=Ind, etc.
                features.append(f"{prefix}{part}")
            elif not upos_done:
                # First bare word is the UPOS tag
                features.append(f"{prefix}UPOS={part}")
                upos_done = True
            elif part.startswith("frame") and part[5:].lstrip("-").isdigit():
                # Dependency frame label: frame3
                features.append(f"{prefix}frame={part[5:]}")
            # else: ignore unrecognised bare segments

        return features

    # Function word or punctuation
    # If already namespaced (en::the), keep as-is inside word=...
    return [f"word={token}"]


# ---------------------------------------------------------------------------
# FactorVocab
# ---------------------------------------------------------------------------

class FactorVocab:
    """
    Bidirectional mappings between tokens, binary features, and their IDs.

    Feature IDs are 1-indexed; 0 is reserved as the padding index so that a
    zero-padded feature list maps to a zero embedding.
    """

    def __init__(self) -> None:
        self.feature_to_id: dict[str, int] = {}   # feat_str -> 1-based ID
        self.id_to_feature: dict[int, str]  = {}   # 1-based ID -> feat_str
        self.token_to_id:   dict[str, int]  = {}   # token -> 0-based index
        self.id_to_token:   dict[int, str]  = {}   # 0-based index -> token
        self.token_features: dict[str, list[int]] = {}  # token -> [feat_ids]

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, vocab_path: Path, min_freq: int = 1) -> None:
        """Parse delex_vocab.txt and populate all mappings.

        min_freq applies only to function-word tokens (word=... features).
        Structural features (UPOS, frame, morph) and special tokens are always
        kept regardless of count — they are not singleton noise.
        """
        tokens: list[tuple[str, int]] = []
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                token = parts[0]
                count = int(parts[1]) if len(parts) > 1 else 0
                # Drop rare function words; always keep structural tokens and <eos>
                is_funcword = not token.startswith("[") and token != "<eos>"
                if is_funcword and count < min_freq:
                    continue
                tokens.append((token, count))

        # Two-pass: first collect all features, then assign IDs
        per_token_feats: dict[str, list[str]] = {}
        all_features: list[str] = []
        seen_feats: set[str] = set()

        for token, _ in tokens:
            feats = parse_token_features(token)
            per_token_feats[token] = feats
            for f in feats:
                if f not in seen_feats:
                    seen_feats.add(f)
                    all_features.append(f)

        # Sort features for determinism: special first, then word=, then structural
        def feat_sort_key(f: str) -> tuple[int, str]:
            if f.startswith("special="):
                return (0, f)
            if f.startswith("word="):
                return (1, f)
            return (2, f)

        all_features.sort(key=feat_sort_key)

        # Assign 1-based IDs (0 = padding)
        for feat_id, feat in enumerate(all_features, start=1):
            self.feature_to_id[feat] = feat_id
            self.id_to_feature[feat_id] = feat

        # Assign 0-based token IDs and resolve feature ID lists
        for tok_id, (token, _) in enumerate(tokens):
            self.token_to_id[token] = tok_id
            self.id_to_token[tok_id] = token
            self.token_features[token] = [
                self.feature_to_id[f] for f in per_token_feats[token]
            ]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_token(self, token: str) -> list[int]:
        """Return list of active feature IDs for *token*; empty list if unknown."""
        return self.token_features.get(token, [])

    def encode_sentence(
        self,
        tokens: list[str],
        max_features: int | None = None,
    ) -> list[list[int]]:
        """
        Encode a sentence as a list of active-feature-ID lists.

        If max_features is given, each list is right-padded with 0s to that
        length (for batching with a fixed-size tensor).
        """
        result = [self.encode_token(t) for t in tokens]
        if max_features is not None:
            result = [
                (ids + [0] * max_features)[:max_features]
                for ids in result
            ]
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        data = {
            "feature_to_id":        self.feature_to_id,
            "id_to_feature":        {str(k): v for k, v in self.id_to_feature.items()},
            "token_to_id":          self.token_to_id,
            "id_to_token":          {str(k): v for k, v in self.id_to_token.items()},
            "token_features":       self.token_features,
            "n_features":           len(self.feature_to_id),
            "n_tokens":             len(self.token_to_id),
            "max_features_per_token": max(
                (len(v) for v in self.token_features.values()), default=0
            ),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "FactorVocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        fv = cls()
        fv.feature_to_id  = data["feature_to_id"]
        fv.id_to_feature  = {int(k): v for k, v in data["id_to_feature"].items()}
        fv.token_to_id    = data["token_to_id"]
        fv.id_to_token    = {int(k): v for k, v in data["id_to_token"].items()}
        fv.token_features = data["token_features"]
        return fv

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_structural = sum(
            1 for f in self.feature_to_id
            if not f.startswith("word=") and not f.startswith("special=")
        )
        n_word       = sum(1 for f in self.feature_to_id if f.startswith("word="))
        n_special    = sum(1 for f in self.feature_to_id if f.startswith("special="))
        max_f        = max((len(v) for v in self.token_features.values()), default=0)
        langs = set()
        for feat in self.feature_to_id:
            if "::" in feat and not feat.startswith("word="):
                langs.add(feat.split("::")[0])
        lines = [
            "FactorVocab summary",
            f"  tokens             : {len(self.token_to_id):,}",
            f"  total features     : {len(self.feature_to_id):,}  (1-indexed; 0=pad)",
            f"    structural        : {n_structural:,}  (UPOS, frame, morph feats)",
            f"    word=             : {n_word:,}  (function words)",
            f"    special=          : {n_special:,}  (eos, etc.)",
            f"  languages detected : {sorted(langs)}",
            f"  max feats/token    : {max_f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-language binary feature vocabulary from delex_vocab.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vocab", metavar="FILE", required=True,
        help="Path to delex_vocab.txt (token TAB count per line).",
    )
    parser.add_argument(
        "--output", metavar="FILE", default="factor_vocab.json",
        help="Output path for the factor vocabulary JSON.",
    )
    parser.add_argument(
        "--min-freq", type=int, default=3, metavar="N",
        help="Minimum count for function-word tokens to be included. "
             "Structural tokens (lexeme templates) are always kept.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a quick self-test and print feature decompositions for sample tokens.",
    )
    args = parser.parse_args()

    if args.test:
        samples = [
            "<eos>",
            "[VERB|Mood=Ind|Tense=Past]",
            "[en|VERB|frame3|Mood=Ind|Tense=Past]",
            "[es|NOUN|frame0|Gender=Masc|Number=Sing]",
            "[en|ADJ|Degree=Pos]",
            "en::the",
            "es::de",
            "the",
            ".",
        ]
        print("Feature decomposition test:")
        for tok in samples:
            feats = parse_token_features(tok)
            print(f"  {tok!r:<45} -> {feats}")
        print()

    fv = FactorVocab()
    fv.build(Path(args.vocab), min_freq=args.min_freq)
    print(fv.summary())

    out = Path(args.output)
    fv.save(out)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
