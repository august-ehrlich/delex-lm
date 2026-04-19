#!/usr/bin/env python3
"""
build_surface_dict.py — Build a surface-form distribution table for qualitative
LM evaluation.

For each delexicalized token type (e.g. [en|VERB|frame3|Mood=Ind|Tense=Past])
we collect every surface form seen for it across all UD gold + silver CoNLL-U
files and record their raw counts.  The resulting dict allows the inference
sampler to replace abstract tokens with real words that have the right
POS/morphology/frame profile.

Output: surface_dict.json
  { "[en|VERB|frame3|Mood=Ind|Tense=Past]": {"walked": 4, "ran": 2, ...}, ... }

Usage:
    python build_surface_dict.py \\
        --ud-dir "Universal Dependencies 2.17/ud-treebanks-v2.17/ud-treebanks-v2.17" \\
        --silver-dir silver_v2_output \\
        --profile-dir profiles \\
        --langs ar,de,en,es,eu,fi,hi,hy,id,ja,ko,lv,ru,tr,zh \\
        --output surface_dict.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from delexicalize import (
    EXCLUDED_TREEBANKS,
    LEXEME_UPOS,
    get_iso_code,
    load_profiles,
    lookup_profile,
    make_delex_token,
    parse_conllu,
)


def scan_conllu(
    filepath: Path,
    lang_code: str,
    profiles: dict | None,
    surface_counts: dict,
) -> int:
    """
    Scan one CoNLL-U file and accumulate surface form counts into surface_counts.
    Returns number of lexeme token observations added.
    """
    added = 0
    for _, rows in parse_conllu(filepath):
        for row in rows:
            if len(row) < 8:
                continue
            form  = row[1]
            lemma = row[2]
            upos  = row[3]
            morph = row[5]

            if upos not in LEXEME_UPOS:
                continue
            if not form or form == "_":
                continue

            profile = lookup_profile(profiles, lemma, upos) if profiles else ""
            key = make_delex_token(upos, morph, lang_code, profile)
            surface_counts[key][form] += 1
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build surface-form distribution table for qualitative LM eval.",
    )
    parser.add_argument("--ud-dir",      required=True, metavar="DIR",
                        help="UD treebanks root (contains UD_*-* subdirs)")
    parser.add_argument("--silver-dir",  default=None,  metavar="DIR",
                        help="Silver data root (contains {lang}/ subdirs)")
    parser.add_argument("--profile-dir", default="profiles", metavar="DIR",
                        help="Dependency profile dir (contains {lang}/profiles_b_k8.tsv)")
    parser.add_argument("--n-clusters",  type=int, default=8,
                        help="K used in profile filename")
    parser.add_argument("--langs",       default="ar,de,en,es,eu,fi,hi,hy,id,ja,ko,lv,ru,tr,zh",
                        help="Comma-separated ISO codes to include")
    parser.add_argument("--output",      default="surface_dict.json", metavar="FILE")
    args = parser.parse_args()

    target_langs = set(args.langs.split(","))
    ud_dir       = Path(args.ud_dir)
    silver_dir   = Path(args.silver_dir) if args.silver_dir else None
    profile_dir  = Path(args.profile_dir)

    # surface_counts[delex_token][surface_form] = int
    surface_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ----------------------------------------------------------------
    # UD gold treebanks
    # ----------------------------------------------------------------
    print("Scanning UD gold treebanks...")
    profile_cache: dict[str, dict] = {}
    ud_total = 0

    for tb_dir in sorted(ud_dir.iterdir()):
        if not tb_dir.is_dir() or not tb_dir.name.startswith("UD_"):
            continue
        tb_name = tb_dir.name
        if tb_name in EXCLUDED_TREEBANKS:
            continue

        iso = get_iso_code(tb_dir)
        if not iso or iso not in target_langs:
            continue

        if iso not in profile_cache:
            profile_cache[iso] = load_profiles(
                str(profile_dir), iso, "b", n_clusters=args.n_clusters
            )
        profiles = profile_cache[iso] or None

        for conllu in sorted(tb_dir.glob("*.conllu")):
            n = scan_conllu(conllu, iso, profiles, surface_counts)
            ud_total += n
            print(f"  {tb_name}/{conllu.name}: {n:,} lexeme tokens")

    print(f"UD gold total: {ud_total:,} lexeme token observations\n")

    # ----------------------------------------------------------------
    # Silver data
    # ----------------------------------------------------------------
    if silver_dir and silver_dir.is_dir():
        print("Scanning silver data...")
        silver_total = 0
        for lang_dir in sorted(silver_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            iso = lang_dir.name
            if iso not in target_langs:
                continue

            if iso not in profile_cache:
                profile_cache[iso] = load_profiles(
                    str(profile_dir), iso, "b", n_clusters=args.n_clusters
                )
            profiles = profile_cache[iso] or None

            lang_n = 0
            for conllu in sorted(lang_dir.glob("*.conllu")):
                lang_n += scan_conllu(conllu, iso, profiles, surface_counts)
            silver_total += lang_n
            print(f"  [{iso}] {lang_n:,} lexeme token observations from silver")
        print(f"Silver total: {silver_total:,} lexeme token observations\n")

    # ----------------------------------------------------------------
    # Summary and save
    # ----------------------------------------------------------------
    n_tokens = len(surface_counts)
    n_forms  = sum(len(v) for v in surface_counts.values())
    avg_forms = n_forms / n_tokens if n_tokens else 0

    print(f"Unique delex token types with surface forms: {n_tokens:,}")
    print(f"Total (token, form) pairs                  : {n_forms:,}")
    print(f"Average surface forms per token type       : {avg_forms:.1f}")

    # Convert nested defaultdicts to plain dicts for JSON serialisation
    out = {k: dict(v) for k, v in surface_counts.items()}

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nSaved -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
