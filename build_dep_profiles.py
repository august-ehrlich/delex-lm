#!/usr/bin/env python3
"""
build_dep_profiles.py — Collect per-lemma dependency observations from
gold UD treebanks + silver UDPipe data.

For each LEXEME_UPOS token in each sentence, records:
  - is_<deprel>_{left,right}: how this token attaches to its head
    (direction = position of token relative to head)
  - takes_<deprel>_{left,right}: what dependents it attracts
    (direction = position of dependent relative to this token)
  - content_valency: count of non-functional dependents per occurrence
    (functional = punct, det, case, mark, cc, aux, cop, expl)

Deprel subtypes (e.g. acl:relcl): by default, stripped to base type (acl).
Use --keep-subtypes to encode as acl_relcl instead.

Output: profiles/{lang}/observations.pkl

Usage:
    python build_dep_profiles.py --lang en \\
        --ud-dir "Universal Dependencies 2.17/ud-treebanks-v2.17/ud-treebanks-v2.17" \\
        --silver-dir silver_v2_output \\
        --output-dir profiles
    python build_dep_profiles.py --all-langs ...
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from delexicalize import LEXEME_UPOS, get_iso_code, parse_conllu

# Dependents that contribute to grammatical structure but not lexical valency
FUNCTIONAL_RELS = {"punct", "det", "case", "mark", "cc", "aux", "cop", "expl"}


# ---------------------------------------------------------------------------
# Deprel normalisation
# ---------------------------------------------------------------------------

def norm_deprel(deprel: str, keep_subtypes: bool) -> str:
    """
    Normalise a deprel string.

    keep_subtypes=False (default): strip subtype  -> "acl:relcl" -> "acl"
    keep_subtypes=True:            encode subtype -> "acl:relcl" -> "acl_relcl"
    """
    if keep_subtypes:
        return deprel.replace(":", "_")
    return deprel.split(":")[0]


# ---------------------------------------------------------------------------
# Per-file observation collection
# ---------------------------------------------------------------------------

def collect_from_file(filepath: Path, obs: dict, keep_subtypes: bool) -> int:
    """
    Stream one CoNLL-U file and accumulate observations into obs.

    obs[lemma][upos] = {
        'n':               int,
        'is_counts':       Counter,   # is_<deprel>_{left,right} or is_root
        'takes_counts':    Counter,   # takes_<deprel>_{left,right}
        'content_valency': Counter,   # {0: n, 1: n, 2: n, ...}
    }

    Returns number of sentences processed.
    """
    n_sents = 0

    for _sent_text, rows in parse_conllu(filepath):
        n_sents += 1

        # Build token map: CoNLL-U 1-based id -> (lemma, upos, deprel, head_id, pos0)
        # pos0 = 0-based sequential position in sentence (for direction arithmetic)
        tokens: dict[int, tuple] = {}
        for pos0, row in enumerate(rows):
            if len(row) < 8:
                continue
            try:
                tok_id = int(row[0])
                head_id = int(row[6])
            except ValueError:
                continue
            tokens[tok_id] = (
                row[2].lower(),   # lemma
                row[3],           # upos
                row[7],           # deprel
                head_id,
                pos0,
            )

        # dependents map: head_id -> [(dep_tok_id, dep_deprel)]
        deps: dict[int, list] = defaultdict(list)
        for tok_id, (_, _, deprel, head_id, _) in tokens.items():
            deps[head_id].append((tok_id, deprel))

        # Collect observations for lexeme tokens
        for tok_id, (lemma, upos, deprel, head_id, pos0) in tokens.items():
            if upos not in LEXEME_UPOS:
                continue

            entry = obs[lemma][upos]
            entry["n"] += 1

            # --- is-feature (how this token attaches to its head) ---
            nd = norm_deprel(deprel, keep_subtypes)
            if head_id == 0:
                entry["is_counts"]["is_root"] += 1
            elif head_id in tokens:
                head_pos0 = tokens[head_id][4]
                if pos0 < head_pos0:
                    entry["is_counts"][f"is_{nd}_left"] += 1
                else:
                    entry["is_counts"][f"is_{nd}_right"] += 1

            # --- takes-features + content valency ---
            content_val = 0
            for dep_id, dep_deprel in deps.get(tok_id, []):
                nd_dep = norm_deprel(dep_deprel, keep_subtypes)
                if dep_id in tokens:
                    dep_pos0 = tokens[dep_id][4]
                    if dep_pos0 < pos0:
                        entry["takes_counts"][f"takes_{nd_dep}_left"] += 1
                    else:
                        entry["takes_counts"][f"takes_{nd_dep}_right"] += 1
                # Content valency uses base deprel (even if keep_subtypes)
                base_rel = dep_deprel.split(":")[0]
                if base_rel not in FUNCTIONAL_RELS:
                    content_val += 1

            entry["content_valency"][content_val] += 1

    return n_sents


# ---------------------------------------------------------------------------
# Per-language processing
# ---------------------------------------------------------------------------

def process_language(
    lang: str,
    ud_dir: Path,
    silver_dir: Path,
    output_dir: Path,
    keep_subtypes: bool,
) -> None:
    out_lang = output_dir / lang
    out_lang.mkdir(parents=True, exist_ok=True)

    # obs[lemma][upos] = dict with n / is_counts / takes_counts / content_valency
    obs: dict = defaultdict(lambda: defaultdict(lambda: {
        "n": 0,
        "is_counts": Counter(),
        "takes_counts": Counter(),
        "content_valency": Counter(),
    }))

    total_files = total_sents = 0

    # --- Gold UD treebanks ---
    if ud_dir.exists():
        for tb_dir in sorted(ud_dir.glob("UD_*-*")):
            if not tb_dir.is_dir():
                continue
            if get_iso_code(tb_dir) != lang:
                continue
            for conllu_file in sorted(tb_dir.glob("*.conllu")):
                n = collect_from_file(conllu_file, obs, keep_subtypes)
                print(f"  [gold  ] {tb_dir.name}/{conllu_file.name}: {n:,} sents")
                total_files += 1
                total_sents += n

    # --- Silver data ---
    silver_lang = silver_dir / lang
    if silver_lang.exists():
        files = sorted(silver_lang.glob("*.conllu"))
        for conllu_file in files:
            n = collect_from_file(conllu_file, obs, keep_subtypes)
            total_files += 1
            total_sents += n
        if files:
            print(f"  [silver] {len(files)} files from {silver_lang}: {total_sents:,} sents total")

    if total_files == 0:
        print(f"  WARNING: no data found for lang='{lang}'")
        return

    # Convert to plain dict for pickling (defaultdict lambdas don't pickle cleanly)
    obs_plain: dict = {
        lemma: {
            upos: {
                "n": e["n"],
                "is_counts": dict(e["is_counts"]),
                "takes_counts": dict(e["takes_counts"]),
                "content_valency": dict(e["content_valency"]),
            }
            for upos, e in upos_dict.items()
        }
        for lemma, upos_dict in obs.items()
    }

    out_pkl = out_lang / "observations.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(obs_plain, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_pairs = sum(len(v) for v in obs_plain.values())
    print(
        f"\n[{lang}] {total_files} files | {total_sents:,} sents | "
        f"{len(obs_plain):,} lemmas | {n_pairs:,} (lemma,upos) pairs"
    )
    print(f"  -> {out_pkl}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-lemma dependency observations from UD gold + silver data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", metavar="ISO",
                        help="Language ISO code (e.g. 'en'). Mutually exclusive with --all-langs.")
    parser.add_argument("--all-langs", action="store_true",
                        help="Process every language found in --silver-dir.")
    parser.add_argument(
        "--ud-dir", metavar="DIR",
        default="Universal Dependencies 2.17/ud-treebanks-v2.17/ud-treebanks-v2.17",
        help="Root directory containing UD_*-* treebank subdirectories.",
    )
    parser.add_argument("--silver-dir", metavar="DIR", default="silver_v2_output",
                        help="Root directory of silver data (subdirs = ISO codes).")
    parser.add_argument("--output-dir", metavar="DIR", default="profiles",
                        help="Root output directory; lang/ subdirs are created automatically.")
    parser.add_argument(
        "--keep-subtypes", action="store_true",
        help="Keep deprel subtypes encoded as underscores (acl:relcl -> acl_relcl). "
             "Default: strip subtypes (acl:relcl -> acl).",
    )
    args = parser.parse_args()

    ud_dir = Path(args.ud_dir)
    silver_dir = Path(args.silver_dir)
    output_dir = Path(args.output_dir)

    if args.all_langs:
        if not silver_dir.exists():
            raise SystemExit(f"Silver dir not found: {silver_dir}")
        langs = sorted(p.name for p in silver_dir.iterdir() if p.is_dir())
    elif args.lang:
        langs = [args.lang]
    else:
        parser.error("Provide --lang ISO or --all-langs.")

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  Language: {lang}")
        print(f"{'='*60}")
        process_language(lang, ud_dir, silver_dir, output_dir, args.keep_subtypes)


if __name__ == "__main__":
    main()
