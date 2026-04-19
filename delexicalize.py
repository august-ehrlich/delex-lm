#!/usr/bin/env python3
"""
delexicalize.py — Multilingual delexicalizing tokenizer for UD treebanks.

Replaces lexemes (NOUN, VERB, ADJ, ADV, PROPN, NUM) with bracket-encoded
POS+morphology tokens like [NOUN|Number=Sing].

With --include-lang-code the ISO 639 language code is prepended to each
masked token so that e.g. English and French singular nouns get distinct
tokens: [en|NOUN|Number=Sing] vs [fr|NOUN|Gender=Masc|Number=Sing].
Without the flag the tokens are language-neutral.

Treebank selection (all thresholds configurable):
  --min-pos    Distinct UPOS tags a treebank must use          (default 8)
  --min-feats  Distinct morphological feature keys             (default 10)
  --min-tokens Total tokens per language (qualifying treebanks only)
                                                               (default 100000)

Treebanks whose language name is a compound of two different language names
(e.g. UD_Telugu_English-TECT) are excluded as code-switching corpora.

Usage:
    python delexicalize.py --ud-dir "Universal Dependencies 2.17" --output-dir output/
    python delexicalize.py --ud-dir ... --output-dir ... --include-lang-code
    python delexicalize.py --test [--include-lang-code]
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEXEME_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}

# Words that can appear as the first component of a compound language name
# WITHOUT indicating code-switching (they are modifier words, not languages).
LANG_MODIFIERS = {
    "Old", "Classical", "Ancient", "Medieval", "Middle", "Modern",
    "Northern", "Southern", "Eastern", "Western", "Upper", "Lower",
    "Central", "New", "Proto",
}

# Known code-switching treebanks not caught by the heuristic
CODESWITCHING_TREEBANKS = {
    "UD_Maghrebi_Arabic_French-Arabizi",
}

# Treebanks excluded for reasons other than code-switching.
# UD_Arabic-NYUAD:    FORM withheld from public release; all non-MWT tokens have
#                     FORM=_ making function-word recovery impossible.
# UD_Korean-Kaist:    Fuses postpositions into NOUN FORM tokens; incompatible
#                     with our function-word extraction approach.
# UD_Japanese-GSDLUW: Word-level (LUW) duplicate of GSD; shares sentence text so
#                     deduplication would collapse both — keeping GSD (morpheme-level).
# UD_Telugu-MTG:      Zero FEATS and LEMMA on all tokens; language excluded entirely.
EXCLUDED_TREEBANKS = {
    "UD_Arabic-NYUAD",
    "UD_Korean-Kaist",
    "UD_Japanese-GSDLUW",
    "UD_Telugu-MTG",
}


# ---------------------------------------------------------------------------
# Treebank language helpers
# ---------------------------------------------------------------------------

def is_codeswitching(tb_name: str) -> bool:
    """
    Return True if the treebank is a code-switching corpus.

    Heuristic: the language part of the directory name (between 'UD_' and '-')
    has two or more components separated by underscores where at least two of
    them look like independent language names (capitalised, not a modifier word).

    Examples:
        UD_Telugu_English-TECT   → True   (Telugu + English)
        UD_Turkish_English-BUTR  → True
        UD_Old_English-Cairo     → False  (Old is a modifier)
        UD_Northern_Sami-Giella  → False  (Northern is a modifier)
    """
    if tb_name in CODESWITCHING_TREEBANKS:
        return True
    lang_part = tb_name[3:].split("-")[0]   # e.g. "Telugu_English"
    components = lang_part.split("_")
    if len(components) < 2:
        return False
    lang_words = [c for c in components if c and c[0].isupper() and c not in LANG_MODIFIERS]
    return len(lang_words) >= 2


def get_iso_code(tb_dir: Path) -> str | None:
    """
    Extract the ISO 639 language code from the prefix of the first .conllu
    filename in the treebank directory.

    e.g.  fr_gsd-ud-train.conllu  →  "fr"
          zh_gsd-ud-train.conllu  →  "zh"
          orv_rnc-ud-test.conllu  →  "orv"
    """
    for f in tb_dir.glob("*.conllu"):
        return f.stem.split("_")[0]
    return None


# ---------------------------------------------------------------------------
# stats.xml parsing
# ---------------------------------------------------------------------------

def parse_stats_xml(tb_dir: Path) -> tuple[int, int, int] | None:
    """
    Parse the treebank's stats.xml and return
        (total_tokens, n_upos_tags, n_feat_keys)
    or None if no stats.xml is present.

    total_tokens  — surface token count across all splits
    n_upos_tags   — number of distinct UPOS tags used  (<tags unique="N">)
    n_feat_keys   — number of distinct morphological feature keys
                    (e.g. Number, Tense, Case — NOT individual feat=val pairs)
    """
    stats_path = tb_dir / "stats.xml"
    if not stats_path.exists():
        return None

    try:
        root = ET.parse(stats_path).getroot()
    except ET.ParseError:
        return None

    # Token count
    tokens_el = root.find(".//total/tokens")
    if tokens_el is None:
        return None
    total_tokens = int(tokens_el.text)

    # UPOS tag count
    tags_el = root.find("tags")
    n_upos = int(tags_el.get("unique", 0)) if tags_el is not None else 0

    # Feature key count (unique `name` attributes, NOT unique feat=val pairs)
    feats_el = root.find("feats")
    if feats_el is not None:
        feat_keys = {feat.get("name") for feat in feats_el.findall("feat")}
        n_feat_keys = len(feat_keys)
    else:
        n_feat_keys = 0

    return total_tokens, n_upos, n_feat_keys


# ---------------------------------------------------------------------------
# Treebank discovery and filtering
# ---------------------------------------------------------------------------

def find_and_filter_treebanks(
    ud_dir: str,
    min_pos: int,
    min_feats: int,
    min_tokens: int,
    force_langs: set[str] | None = None,
) -> tuple[dict, list]:
    """
    Discover all UD treebanks under ud_dir, apply per-treebank and per-language
    filtering criteria, and return qualifying treebanks grouped by ISO code.

    Algorithm:
      1. Scan every UD_*-* directory.
      2. Skip code-switching treebanks.
      3. Keep treebanks with n_upos >= min_pos AND n_feat_keys >= min_feats.
         Exception: languages in force_langs bypass the min_feats check
         (for annotation-style reasons, e.g. Japanese, Korean).
      4. Group survivors by ISO language code; sum their token counts.
      5. Keep languages whose qualifying token total >= min_tokens.

    Returns:
        qualifying : {iso_code: {tb_name: {"splits": {split: Path},
                                           "tokens": int}}}
        report     : list of dicts describing every treebank considered
                     (useful for printing a summary of what was kept/dropped)
    """
    ud_path = Path(ud_dir)
    seen_names: set[str] = set()

    # {iso_code: {tb_name: {"splits": ..., "tokens": ...}}}
    candidates: dict[str, dict] = defaultdict(dict)
    report: list[dict] = []

    for tb_dir in sorted(ud_path.glob("**/UD_*-*")):
        if not tb_dir.is_dir():
            continue
        tb_name = tb_dir.name
        if tb_name in seen_names:
            continue
        seen_names.add(tb_name)

        # Code-switching check
        if is_codeswitching(tb_name):
            report.append({"tb": tb_name, "status": "excluded (code-switching)"})
            continue

        # Explicit exclusion list (e.g. treebanks missing surface forms)
        if tb_name in EXCLUDED_TREEBANKS:
            report.append({"tb": tb_name, "status": "excluded (explicitly excluded)"})
            continue

        iso_code = get_iso_code(tb_dir)
        if not iso_code:
            report.append({"tb": tb_name, "status": "excluded (no conllu files)"})
            continue

        stats = parse_stats_xml(tb_dir)
        if stats is None:
            report.append({"tb": tb_name, "iso": iso_code, "status": "excluded (no stats.xml)"})
            continue

        total_tokens, n_upos, n_feat_keys = stats

        if n_upos < min_pos:
            report.append({
                "tb": tb_name, "iso": iso_code, "tokens": total_tokens,
                "n_upos": n_upos, "n_feats": n_feat_keys,
                "status": f"excluded (only {n_upos} POS < {min_pos})",
            })
            continue

        if n_feat_keys < min_feats and (force_langs is None or iso_code not in force_langs):
            report.append({
                "tb": tb_name, "iso": iso_code, "tokens": total_tokens,
                "n_upos": n_upos, "n_feats": n_feat_keys,
                "status": f"excluded (only {n_feat_keys} feat keys < {min_feats})",
            })
            continue

        # Collect splits
        splits: dict[str, Path] = {}
        for f in sorted(tb_dir.glob("*.conllu")):
            stem = f.stem
            if "-train" in stem:
                splits["train"] = f
            elif "-dev" in stem:
                splits["dev"] = f
            elif "-test" in stem:
                splits["test"] = f

        if not splits:
            report.append({"tb": tb_name, "iso": iso_code, "status": "excluded (no splits found)"})
            continue

        candidates[iso_code][tb_name] = {"splits": splits, "tokens": total_tokens,
                                          "n_upos": n_upos, "n_feats": n_feat_keys}
        report.append({
            "tb": tb_name, "iso": iso_code, "tokens": total_tokens,
            "n_upos": n_upos, "n_feats": n_feat_keys, "status": "candidate",
        })

    # Filter by per-language token total
    qualifying: dict[str, dict] = {}
    for iso_code, treebanks in candidates.items():
        lang_tokens = sum(tb["tokens"] for tb in treebanks.values())
        if lang_tokens >= min_tokens:
            qualifying[iso_code] = treebanks
        else:
            for tb_name in treebanks:
                for entry in report:
                    if entry.get("tb") == tb_name and entry.get("status") == "candidate":
                        entry["status"] = (
                            f"excluded (language total {lang_tokens:,} < {min_tokens:,})"
                        )

    # Mark survivors in report
    for iso_code, treebanks in qualifying.items():
        for tb_name in treebanks:
            for entry in report:
                if entry.get("tb") == tb_name and entry.get("status") == "candidate":
                    entry["status"] = "included"

    return qualifying, report


# ---------------------------------------------------------------------------
# Dependency profile loading
# ---------------------------------------------------------------------------

def load_profiles(profile_dir: str, lang_code: str, mode: str, n_clusters: int = 20) -> dict:
    """
    Load a dependency profile lookup table for a language.

    Returns a dict: (lemma_lower, upos) → profile_str
      mode 'a': profile_str = pipe-joined active feature names, e.g. "takes_obj_right|is_nsubj_right"
      mode 'b': profile_str = "frame{cluster_id}", e.g. "frame3"

    Fallbacks:
      - lemma not found → look up upos-level default (precomputed centroid per UPOS)
      - upos not found either → empty string (no profile appended)
    """
    base = Path(profile_dir) / lang_code
    profiles: dict[tuple[str, str], str] = {}

    if mode == "a":
        tsv = base / "profiles_a.tsv"
        meta_path = base / "profiles_a_meta.json"
        if not tsv.exists():
            return {}
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        feature_names = meta["feature_names"]
        with open(tsv, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                cols = line.rstrip("\n").split("\t")
                lemma, upos = cols[0], cols[1]
                active = [feature_names[i] for i, v in enumerate(cols[3:]) if v == "1"]
                profiles[(lemma, upos)] = "|".join(active)

        # UPOS-level defaults: OR of all profiles for that UPOS (majority vote per feature)
        upos_feat_sums: dict[str, defaultdict] = {}
        upos_counts: dict[str, int] = defaultdict(int)
        with open(tsv, encoding="utf-8") as f:
            next(f)
            for line in f:
                cols = line.rstrip("\n").split("\t")
                upos = cols[1]
                upos_counts[upos] += 1
                if upos not in upos_feat_sums:
                    upos_feat_sums[upos] = defaultdict(int)
                for i, v in enumerate(cols[3:]):
                    if v == "1":
                        upos_feat_sums[upos][feature_names[i]] += 1
        threshold = meta.get("threshold", 0.33)
        for upos, sums in upos_feat_sums.items():
            n = upos_counts[upos]
            active = [f for f, s in sums.items() if s / n >= threshold]
            profiles[("__default__", upos)] = "|".join(active)

    elif mode == "b":
        tsv = base / f"profiles_b_k{n_clusters}.tsv"
        meta_path = base / f"profiles_b_k{n_clusters}_meta.json"
        if not tsv.exists():
            return {}
        with open(tsv, encoding="utf-8") as f:
            next(f)
            for line in f:
                cols = line.rstrip("\n").split("\t")
                lemma, upos, cluster_id = cols[0], cols[1], cols[3]
                profiles[(lemma, upos)] = f"frame{cluster_id}"

        # UPOS-level defaults: most common cluster per UPOS
        upos_clusters: dict[str, list] = defaultdict(list)
        for (lemma, upos), profile_str in profiles.items():
            if lemma != "__default__":
                upos_clusters[upos].append(profile_str)
        for upos, cluster_list in upos_clusters.items():
            most_common = max(set(cluster_list), key=cluster_list.count)
            profiles[("__default__", upos)] = most_common

    return profiles


def lookup_profile(profiles: dict, lemma: str, upos: str) -> str:
    """
    Look up profile string for (lemma, upos).
    Falls back to UPOS-level default, then empty string.
    """
    key = (lemma.lower(), upos)
    if key in profiles:
        return profiles[key]
    return profiles.get(("__default__", upos), "")


# ---------------------------------------------------------------------------
# Delexicalized token format
# ---------------------------------------------------------------------------

def make_delex_token(
    upos: str,
    morph: str,
    lang_code: str | None = None,
    profile: str = "",
) -> str:
    """
    Build a delexicalized token string.

    Without lang_code:  [NOUN|Number=Sing]           or  [NOUN]
    With lang_code:     [en|NOUN|Number=Sing]         or  [en|NOUN]
    With profile:       [VERB|frame3|Mood=Ind|...]    (profile inserted after UPOS)
    """
    profile_str = f"|{profile}" if profile else ""
    feat_str = ""
    if morph and morph != "_":
        feat_str = "|" + "|".join(sorted(morph.split("|")))

    if lang_code:
        return f"[{lang_code}|{upos}{profile_str}{feat_str}]"
    return f"[{upos}{profile_str}{feat_str}]"


# ---------------------------------------------------------------------------
# CoNLL-U parsing
# ---------------------------------------------------------------------------

def parse_conllu(filepath: Path):
    """
    Yield (sentence_text_or_None, rows) for each sentence in a CoNLL-U file.
    Skips comment lines, multi-word token lines (e.g. 1-2), and empty nodes (1.1).
    """
    sent_text = None
    rows: list[list[str]] = []

    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("# text = "):
                sent_text = line[9:]
            elif line.startswith("#"):
                continue
            elif line == "":
                if rows:
                    yield sent_text, rows
                sent_text = None
                rows = []
            else:
                cols = line.split("\t")
                idx = cols[0]
                if "-" in idx or "." in idx:
                    continue
                rows.append(cols)

    if rows:
        yield sent_text, rows


# ---------------------------------------------------------------------------
# Sentence-level processing
# ---------------------------------------------------------------------------

def delexicalize(
    rows: list[list[str]],
    lang_code: str | None = None,
    profiles: dict | None = None,
) -> str:
    """
    Convert CoNLL-U rows to a delexicalized sentence string.

    Lexemes (LEXEME_UPOS)  → [UPOS|features]  (or [lang|UPOS|features])
    All other tokens       → lowercased surface form  (or lang::word)

    Tokens are always separated by spaces regardless of SpaceAfter=No.
    Surface-form spacing is irrelevant for LM training and ignoring it
    prevents CoNLL-U SpaceAfter=No chains (e.g. date ranges like 2014-2015)
    from being concatenated into a single spurious vocabulary entry.

    Column 2 (LEMMA) is used for profile lookup when profiles is not None.
    """
    toks: list[str] = []
    for row in rows:
        if len(row) < 10:
            continue
        word, lemma, upos, morph = row[1], row[2], row[3], row[5]
        if upos in LEXEME_UPOS:
            profile = lookup_profile(profiles, lemma, upos) if profiles else ""
            toks.append(make_delex_token(upos, morph, lang_code, profile))
        else:
            toks.append(f"{lang_code}::{word.lower()}" if lang_code else word.lower())
    return " ".join(toks)


def surface_text(rows: list[list[str]]) -> str:
    """Reconstruct original surface text for use as a deduplication key."""
    parts: list[tuple[str, bool]] = []
    for row in rows:
        if len(row) < 10:
            continue
        parts.append((row[1], "SpaceAfter=No" in row[9]))
    out = ""
    for i, (word, no_space) in enumerate(parts):
        out += word
        if not no_space and i < len(parts) - 1:
            out += " "
    return out


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    print(f"Scanning treebanks (min_pos={args.min_pos}, "
          f"min_feats={args.min_feats}, min_tokens={args.min_tokens:,})...")

    # Parse target language filter
    target_langs: set[str] | None = None
    if args.langs:
        target_langs = {l.strip() for l in args.langs.split(",")}
        print(f"Language filter: {sorted(target_langs)}")

    # Pre-load profiles if requested
    profile_cache: dict[str, dict] = {}  # iso_code → profiles dict
    if args.profile_dir and args.profile_mode:
        print(f"Profile mode: {args.profile_mode}  dir: {args.profile_dir}")

    qualifying, report = find_and_filter_treebanks(
        args.ud_dir, args.min_pos, args.min_feats, args.min_tokens,
        force_langs=target_langs,
    )

    # Filter to target languages if specified
    if target_langs:
        qualifying = {iso: tbs for iso, tbs in qualifying.items() if iso in target_langs}

    if not qualifying:
        raise SystemExit("No treebanks passed the filtering criteria.")

    # Print selection summary
    included = {e["tb"] for e in report if e.get("status") == "included"}
    excluded = [e for e in report if e.get("status", "").startswith("excluded")]

    print(f"\nIncluded: {len(included)} treebanks across {len(qualifying)} languages")
    for iso_code in sorted(qualifying):
        tbs = qualifying[iso_code]
        lang_tokens = sum(tb["tokens"] for tb in tbs.values())
        tb_strs = ", ".join(
            f"{n}({tb['n_upos']}pos/{tb['n_feats']}feats)"
            for n, tb in sorted(tbs.items())
        )
        print(f"  [{iso_code}] {lang_tokens:>9,} tokens — {tb_strs}")

    print(f"\nExcluded: {len(excluded)} treebanks")
    for e in excluded:
        print(f"  {e['tb']}: {e['status']}")

    lang_code_flag = args.include_lang_code
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Open per-treebank output files
    tb_handles: dict[str, dict[str, object]] = {}
    for iso_code, treebanks in qualifying.items():
        for tb_name, tb_info in treebanks.items():
            tb_dir = out_path / tb_name
            tb_dir.mkdir(exist_ok=True)
            tb_handles[tb_name] = {
                split: open(tb_dir / f"{split}.txt", "w", encoding="utf-8")
                for split in tb_info["splits"]
            }

    # Per-language output files: langs/{iso}/{split}.txt
    # Gold UD data → train_gold.txt / dev.txt / test.txt
    # Silver fill  → train_silver.txt
    all_langs_for_output = set(qualifying.keys())
    if target_langs:
        all_langs_for_output |= target_langs
    langs_dir = out_path / "langs"
    lang_handles: dict[str, dict[str, object]] = {}
    for iso_code in all_langs_for_output:
        lang_out = langs_dir / iso_code
        lang_out.mkdir(parents=True, exist_ok=True)
        lang_handles[iso_code] = {
            "train_gold":   open(lang_out / "train_gold.txt",   "w", encoding="utf-8"),
            "train_silver": open(lang_out / "train_silver.txt", "w", encoding="utf-8"),
            "dev":          open(lang_out / "dev.txt",           "w", encoding="utf-8"),
            "test":         open(lang_out / "test.txt",          "w", encoding="utf-8"),
        }

    # Aggregate files across all languages
    agg = {
        split: open(out_path / f"all_{split}.txt", "w", encoding="utf-8")
        for split in ["train", "dev", "test"]
    }

    # Deduplication: keyed by (iso_code, sent_text) to dedup within a language
    # but allow the same structural sentence to appear in multiple languages.
    seen: set[tuple[str, str]] = set()
    vocab: dict[str, int] = defaultdict(int)
    split_tokens: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    lang_train_tokens: dict[str, int] = defaultdict(int)  # for per-lang cap tracking
    # Track gold/silver token counts per language for dataset_info.json
    lang_gold_tokens:   dict[str, int] = defaultdict(int)
    lang_silver_tokens: dict[str, int] = defaultdict(int)
    total_written = total_dupes = 0

    max_per_lang = args.max_tokens_per_lang  # 0 means no cap

    def write_sentence(line: str, iso_code: str, split: str,
                       tb_handle=None, source: str = "gold") -> None:
        """Write a sentence to treebank file (if given), per-lang file, and aggregate."""
        nonlocal total_written
        if tb_handle is not None:
            tb_handle.write(line)
        # Per-language file
        if iso_code in lang_handles:
            if split == "train":
                lang_key = "train_gold" if source == "gold" else "train_silver"
            else:
                lang_key = split  # "dev" or "test"
            lang_handles[iso_code][lang_key].write(line)
        agg[split].write(line)
        total_written += 1
        toks = line.split()
        n = len(toks)
        split_tokens[split] += n
        if split == "train":
            lang_train_tokens[iso_code] += n
            if source == "gold":
                lang_gold_tokens[iso_code] += n
            else:
                lang_silver_tokens[iso_code] += n
        for tok in toks:
            vocab[tok] += 1

    print()
    # Process test → dev → train so evaluation splits are never contaminated
    for split in ["test", "dev", "train"]:
        for iso_code in sorted(qualifying):
            for tb_name in sorted(qualifying[iso_code]):
                tb_info = qualifying[iso_code][tb_name]
                if split not in tb_info["splits"]:
                    continue

                filepath = tb_info["splits"][split]
                written = dupes = 0
                lc = iso_code if lang_code_flag else None

                # Load profiles for this language on first encounter
                lang_profiles = None
                if args.profile_dir and args.profile_mode:
                    if iso_code not in profile_cache:
                        profile_cache[iso_code] = load_profiles(
                            args.profile_dir, iso_code, args.profile_mode,
                            n_clusters=args.profile_n_clusters,
                        )
                    lang_profiles = profile_cache[iso_code] or None

                for sent_text, rows in parse_conllu(filepath):
                    # Per-language train cap (dev/test are never capped)
                    if split == "train" and max_per_lang > 0:
                        if lang_train_tokens[iso_code] >= max_per_lang:
                            break

                    key_text = sent_text if sent_text is not None else surface_text(rows)
                    dedup_key = (iso_code, key_text)
                    if dedup_key in seen:
                        dupes += 1
                        continue
                    seen.add(dedup_key)

                    line = delexicalize(rows, lc, lang_profiles) + " <eos>\n"
                    write_sentence(line, iso_code, split,
                                   tb_handle=tb_handles[tb_name].get(split),
                                   source="gold")
                    written += 1

                total_dupes += dupes
                print(f"  [{iso_code}] {tb_name}/{split}: "
                      f"{written} written, {dupes} duplicates skipped"
                      + (f" [{lang_train_tokens[iso_code]:,} train tok]"
                         if split == "train" else ""))

    # -----------------------------------------------------------------------
    # Silver data: fill languages below the per-lang cap (train only)
    # -----------------------------------------------------------------------
    if args.silver_dir and max_per_lang > 0:
        silver_path = Path(args.silver_dir)
        print(f"\nFilling from silver data (cap={max_per_lang:,} tokens/lang)...")
        all_langs = target_langs if target_langs else set(qualifying.keys())
        for iso_code in sorted(all_langs):
            lang_silver_dir = silver_path / iso_code
            if not lang_silver_dir.is_dir():
                continue
            remaining = max_per_lang - lang_train_tokens[iso_code]
            if remaining <= 0:
                print(f"  [{iso_code}] already at cap, skipping silver")
                continue

            lc = iso_code if lang_code_flag else None
            lang_profiles = None
            if args.profile_dir and args.profile_mode:
                if iso_code not in profile_cache:
                    profile_cache[iso_code] = load_profiles(
                        args.profile_dir, iso_code, args.profile_mode,
                        n_clusters=args.profile_n_clusters,
                    )
                lang_profiles = profile_cache[iso_code] or None

            written = dupes = 0
            for silver_file in sorted(lang_silver_dir.glob("*.conllu")):
                if lang_train_tokens[iso_code] >= max_per_lang:
                    break
                for sent_text, rows in parse_conllu(silver_file):
                    if lang_train_tokens[iso_code] >= max_per_lang:
                        break
                    key_text = sent_text if sent_text is not None else surface_text(rows)
                    dedup_key = (iso_code, key_text)
                    if dedup_key in seen:
                        dupes += 1
                        continue
                    seen.add(dedup_key)

                    line = delexicalize(rows, lc, lang_profiles) + " <eos>\n"
                    write_sentence(line, iso_code, "train", tb_handle=None,
                                   source="silver")
                    written += 1

            print(f"  [{iso_code}] silver: {written} written, {dupes} dupes"
                  f" — total train tokens: {lang_train_tokens[iso_code]:,}")

    elif args.silver_dir and max_per_lang == 0:
        # No cap: include all silver in train
        silver_path = Path(args.silver_dir)
        print(f"\nAppending all silver data to train (no cap)...")
        all_langs = target_langs if target_langs else set(qualifying.keys())
        for iso_code in sorted(all_langs):
            lang_silver_dir = silver_path / iso_code
            if not lang_silver_dir.is_dir():
                continue

            lc = iso_code if lang_code_flag else None
            lang_profiles = None
            if args.profile_dir and args.profile_mode:
                if iso_code not in profile_cache:
                    profile_cache[iso_code] = load_profiles(
                        args.profile_dir, iso_code, args.profile_mode,
                        n_clusters=args.profile_n_clusters,
                    )
                lang_profiles = profile_cache[iso_code] or None

            written = dupes = 0
            for silver_file in sorted(lang_silver_dir.glob("*.conllu")):
                for sent_text, rows in parse_conllu(silver_file):
                    key_text = sent_text if sent_text is not None else surface_text(rows)
                    dedup_key = (iso_code, key_text)
                    if dedup_key in seen:
                        dupes += 1
                        continue
                    seen.add(dedup_key)

                    line = delexicalize(rows, lc, lang_profiles) + " <eos>\n"
                    write_sentence(line, iso_code, "train", tb_handle=None,
                                   source="silver")
                    written += 1

            print(f"  [{iso_code}] silver: {written} written, {dupes} dupes")

    for split_files in tb_handles.values():
        for f in split_files.values():
            f.close()
    for lang_files in lang_handles.values():
        for f in lang_files.values():
            f.close()
    for f in agg.values():
        f.close()

    # dataset_info.json: per-language gold/silver token counts
    dataset_info: dict = {}
    for iso_code in sorted(all_langs_for_output):
        dataset_info[iso_code] = {
            "train_gold_tokens":   lang_gold_tokens.get(iso_code, 0),
            "train_silver_tokens": lang_silver_tokens.get(iso_code, 0),
            "train_total_tokens":  lang_train_tokens.get(iso_code, 0),
        }
    info_path = out_path / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Dataset info written to : {info_path}")

    # Vocabulary file
    vocab_path = out_path / "delex_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok, count in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write(f"{tok}\t{count}\n")

    total_tokens = sum(split_tokens.values())
    print(f"\nTotal sentences written : {total_written:,}")
    print(f"Duplicates skipped      : {total_dupes:,}")
    print(f"\nToken counts:")
    print(f"  train : {split_tokens['train']:>12,}")
    print(f"  dev   : {split_tokens['dev']:>12,}")
    print(f"  test  : {split_tokens['test']:>12,}")
    print(f"  total : {total_tokens:>12,}")
    if max_per_lang > 0:
        print(f"\nPer-language train token counts:")
        for iso_code in sorted(lang_train_tokens):
            print(f"  [{iso_code}] {lang_train_tokens[iso_code]:>10,}")
    print(f"\nUnique delex token types: {len(vocab):,}")
    print(f"Language code in tokens : {lang_code_flag}")
    print(f"Vocabulary written to   : {vocab_path}")


# ---------------------------------------------------------------------------
# Tokenization test
# ---------------------------------------------------------------------------

def test_tokenization(include_lang_code: bool) -> None:
    """
    Tokenize a sample EWT sentence and display token-by-token comparison,
    with and without the language code toggle.
    """
    sample_rows = [
        ["1",  "[",         "[",         "PUNCT", "-LRB-", "_",                                                    "10", "punct", "10:punct",   "SpaceAfter=No"],
        ["2",  "This",      "this",      "DET",   "DT",    "Number=Sing|PronType=Dem",                             "3",  "det",   "3:det",      "_"],
        ["3",  "killing",   "killing",   "NOUN",  "NN",    "Number=Sing",                                          "10", "nsubj", "10:nsubj",   "_"],
        ["4",  "of",        "of",        "ADP",   "IN",    "_",                                                    "7",  "case",  "7:case",     "_"],
        ["5",  "a",         "a",         "DET",   "DT",    "Definite=Ind|PronType=Art",                            "7",  "det",   "7:det",      "_"],
        ["6",  "respected", "respected", "ADJ",   "JJ",    "Degree=Pos",                                           "7",  "amod",  "7:amod",     "_"],
        ["7",  "cleric",    "cleric",    "NOUN",  "NN",    "Number=Sing",                                          "3",  "nmod",  "3:nmod:of",  "_"],
        ["8",  "will",      "will",      "AUX",   "MD",    "VerbForm=Fin",                                         "10", "aux",   "10:aux",     "_"],
        ["9",  "be",        "be",        "AUX",   "VB",    "VerbForm=Inf",                                         "10", "aux",   "10:aux",     "_"],
        ["10", "causing",   "cause",     "VERB",  "VBG",   "Tense=Pres|VerbForm=Part",                             "0",  "root",  "0:root",     "_"],
        ["11", "us",        "we",        "PRON",  "PRP",   "Case=Acc|Number=Plur|Person=1|PronType=Prs",           "10", "iobj",  "10:iobj",    "_"],
        ["12", "trouble",   "trouble",   "NOUN",  "NN",    "Number=Sing",                                          "10", "obj",   "10:obj",     "_"],
        ["13", "for",       "for",       "ADP",   "IN",    "_",                                                    "14", "case",  "14:case",    "_"],
        ["14", "years",     "year",      "NOUN",  "NNS",   "Number=Plur",                                          "10", "obl",   "10:obl:for", "_"],
        ["15", "to",        "to",        "PART",  "TO",    "_",                                                    "16", "mark",  "16:mark",    "_"],
        ["16", "come",      "come",      "VERB",  "VB",    "VerbForm=Inf",                                         "14", "acl",   "14:acl:to",  "SpaceAfter=No"],
        ["17", ".",         ".",         "PUNCT", ".",     "_",                                                    "10", "punct", "10:punct",   "SpaceAfter=No"],
        ["18", "]",         "]",         "PUNCT", "-RRB-", "_",                                                    "10", "punct", "10:punct",   "_"],
    ]

    lc = "en" if include_lang_code else None
    original = "[This killing of a respected cleric will be causing us trouble for years to come.]"
    delex    = delexicalize(sample_rows, lc) + " <eos>"

    W, U, M = 12, 7, 46
    sep = "=" * 100
    print(sep)
    print(f"TOKENIZATION TEST  --  EWT train sentence 2  "
          f"(lang code: {'ON (en)' if include_lang_code else 'OFF'})")
    print(sep)
    print(f"\nOriginal:      {original}")
    print(f"Delexicalized: {delex}\n")
    print(f"{'WORD':<{W}} {'UPOS':<{U}} {'MORPH':<{M}} OUTPUT")
    print("-" * 100)
    for row in sample_rows:
        word, upos, morph = row[1], row[3], row[5]
        out = make_delex_token(upos, morph, lc) if upos in LEXEME_UPOS else word.lower()
        tag = "  <- masked" if upos in LEXEME_UPOS else ""
        print(f"{word:<{W}} {upos:<{U}} {morph:<{M}} {out}{tag}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual UD delexicalizer for LM training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ud-dir",       metavar="DIR",
                        help="UD release directory containing ud-treebanks-v2.17/")
    parser.add_argument("--output-dir",   metavar="DIR",
                        help="Directory to write delexicalized output")
    parser.add_argument("--include-lang-code", action="store_true",
                        help="Prepend ISO language code to each masked token: "
                             "[en|NOUN|Number=Sing] instead of [NOUN|Number=Sing]")
    parser.add_argument("--min-pos",    type=int, default=8,
                        help="Min distinct UPOS tags a treebank must use")
    parser.add_argument("--min-feats",  type=int, default=10,
                        help="Min distinct morphological feature keys a treebank must use")
    parser.add_argument("--min-tokens", type=int, default=100_000,
                        help="Min total tokens per language (after per-treebank filtering)")
    parser.add_argument("--test", action="store_true",
                        help="Run built-in tokenization demo and exit")
    parser.add_argument("--profile-dir", metavar="DIR", default=None,
                        help="Directory containing per-language profile files "
                             "(e.g. profiles/). Enables dependency profile features in output.")
    parser.add_argument("--profile-mode", choices=["a", "b"], default=None,
                        help="Profile mode: a = binarised features, b = cluster frame ID.")
    parser.add_argument("--profile-n-clusters", type=int, default=20,
                        help="[profile-mode b] Number of clusters used when profiles were built.")
    parser.add_argument("--langs", metavar="ISO[,ISO...]", default=None,
                        help="Comma-separated ISO codes to process (e.g. en,de,fr). "
                             "If omitted, all qualifying languages are processed.")
    parser.add_argument("--silver-dir", metavar="DIR", default=None,
                        help="Directory containing per-language silver .conllu files "
                             "(e.g. silver_v2_output/). Silver sentences are appended to "
                             "train after UD gold data, up to --max-tokens-per-lang.")
    parser.add_argument("--max-tokens-per-lang", type=int, default=0,
                        help="Cap training tokens per language (UD gold first, then silver). "
                             "0 = no cap.")
    args = parser.parse_args()

    if args.test:
        test_tokenization(args.include_lang_code)
        return

    if not args.ud_dir or not args.output_dir:
        parser.error("--ud-dir and --output-dir are required unless --test is given")

    run(args)


if __name__ == "__main__":
    main()
