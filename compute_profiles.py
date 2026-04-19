#!/usr/bin/env python3
"""
compute_profiles.py — Compute dependency profiles from observations.pkl.

Mode A (--mode a):
  For each (lemma, upos) with n >= min_count, binarise each feature proportion
  at a threshold, and assign a valency bucket.
  Output: profiles/{lang}/profiles_a.tsv + profiles_a_meta.json

Mode B (--mode b):
  For each (lemma, upos) with n >= min_count, build a proportion vector,
  standardise, and cluster with k-means.
  Output: profiles/{lang}/profiles_b_k{K}.tsv + profiles_b_k{K}_meta.json

Usage:
    python compute_profiles.py --lang en --mode a --threshold 0.33 --min-count 20
    python compute_profiles.py --lang en --mode b --n-clusters 20 --min-count 20
    python compute_profiles.py --all-langs --mode a --threshold 0.33 --min-count 20
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared: load observations + build feature matrix
# ---------------------------------------------------------------------------

def load_obs(lang_dir: Path) -> dict:
    pkl = lang_dir / "observations.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Missing {pkl} — run build_dep_profiles.py first."
        )
    with open(pkl, "rb") as f:
        return pickle.load(f)


VAL_BUCKETS = ["val_0", "val_1", "val_2", "val_3plus"]


def build_matrix(obs: dict, min_count: int):
    """
    Collect all qualifying (lemma, upos) pairs, align to a shared feature
    index, and return proportion vectors.

    Returns
    -------
    keys          : list of (lemma, upos, n)
    feature_names : list of str  (is_* sorted, takes_* sorted, val_*)
    X             : np.ndarray shape (N, F) of float32 proportions
    """
    all_is = set()
    all_takes = set()
    keys = []

    for lemma, upos_dict in obs.items():
        for upos, entry in upos_dict.items():
            if entry["n"] < min_count:
                continue
            keys.append((lemma, upos, entry["n"]))
            all_is.update(entry["is_counts"])
            all_takes.update(entry["takes_counts"])

    feature_names = sorted(all_is) + sorted(all_takes) + VAL_BUCKETS
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    N = len(keys)
    F = len(feature_names)
    X = np.zeros((N, F), dtype=np.float32)

    for row_i, (lemma, upos, n) in enumerate(keys):
        entry = obs[lemma][upos]

        for feat, cnt in entry["is_counts"].items():
            if feat in feat_idx:
                X[row_i, feat_idx[feat]] = cnt / n

        for feat, cnt in entry["takes_counts"].items():
            if feat in feat_idx:
                X[row_i, feat_idx[feat]] = cnt / n

        # Valency: dominant content-valency bucket
        cv = entry["content_valency"]
        total_cv = sum(cv.values())
        if total_cv > 0:
            bucket_counts = [
                cv.get(0, 0),
                cv.get(1, 0),
                cv.get(2, 0),
                sum(v for k, v in cv.items() if k >= 3),
            ]
            dominant = int(np.argmax(bucket_counts))
            X[row_i, feat_idx[VAL_BUCKETS[dominant]]] = 1.0

    return keys, feature_names, X


# ---------------------------------------------------------------------------
# Mode A: binarise
# ---------------------------------------------------------------------------

def run_mode_a(lang: str, lang_dir: Path, obs: dict, min_count: int, threshold: float) -> None:
    print(f"  Building feature matrix (min_count={min_count})...")
    keys, feature_names, X = build_matrix(obs, min_count)
    print(f"  {len(keys):,} (lemma,upos) pairs  |  {len(feature_names)} features")

    X_bin = (X >= threshold).astype(np.int8)

    out_tsv = lang_dir / "profiles_a.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(["lemma", "upos", "n"] + feature_names) + "\n")
        for row_i, (lemma, upos, n) in enumerate(keys):
            vals = "\t".join(str(v) for v in X_bin[row_i].tolist())
            f.write(f"{lemma}\t{upos}\t{n}\t{vals}\n")

    meta = {
        "lang": lang,
        "mode": "a",
        "threshold": threshold,
        "min_count": min_count,
        "n_lemma_upos": len(keys),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_per_upos": dict(Counter(upos for _, upos, _ in keys)),
    }
    out_meta = lang_dir / "profiles_a_meta.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  -> {out_tsv}")
    print(f"  -> {out_meta}")


# ---------------------------------------------------------------------------
# Mode B: k-means clustering
# ---------------------------------------------------------------------------

def run_mode_b(lang: str, lang_dir: Path, obs: dict, min_count: int, n_clusters: int) -> None:
    """
    Cluster lemmas within each UPOS independently.

    Each POS gets its own set of cluster IDs 0..K-1, fitted on the subset of
    (lemma, upos) pairs for that POS only.  This ensures cluster 3 for VERB and
    cluster 3 for NOUN are unrelated -- the clustering reflects within-class
    syntactic variation rather than cross-POS structure.

    If a UPOS has fewer qualifying pairs than n_clusters, K is reduced to
    n_pairs - 1 for that POS with a warning.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print(f"  Building feature matrix (min_count={min_count})...")
    keys, feature_names, X = build_matrix(obs, min_count)
    print(f"  {len(keys):,} (lemma,upos) pairs  |  {len(feature_names)} features")

    # Group row indices by UPOS
    upos_rows: dict[str, list[int]] = defaultdict(list)
    for row_i, (_, upos, _) in enumerate(keys):
        upos_rows[upos].append(row_i)

    # Cluster each UPOS independently; collect results into a global labels array
    labels = np.full(len(keys), -1, dtype=np.int32)
    per_upos_meta: dict[str, dict] = {}

    for upos in sorted(upos_rows):
        idxs = upos_rows[upos]
        X_sub = X[idxs]
        k = min(n_clusters, len(idxs) - 1)
        if k < 2:
            labels[idxs] = 0
            per_upos_meta[upos] = {
                "n_clusters_used": 1,
                "n_pairs": len(idxs),
                "cluster_sizes": {0: len(idxs)},
                "cluster_centroids_raw": [X_sub.mean(axis=0).tolist()],
            }
            print(f"    {upos}: {len(idxs)} pairs -- too few for k={n_clusters}, assigned all to cluster 0")
            continue

        if k < n_clusters:
            print(f"    {upos}: {len(idxs)} pairs -- reducing k to {k}")

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sub_labels = km.fit_predict(X_sub_scaled)
        for local_i, global_i in enumerate(idxs):
            labels[global_i] = int(sub_labels[local_i])

        cluster_sizes = dict(Counter(int(l) for l in sub_labels))
        raw_centroids = [
            X_sub[sub_labels == c].mean(axis=0).tolist() if (sub_labels == c).sum() > 0
            else [0.0] * len(feature_names)
            for c in range(k)
        ]
        per_upos_meta[upos] = {
            "n_clusters_used": k,
            "n_pairs": len(idxs),
            "cluster_sizes": cluster_sizes,
            "cluster_centroids_raw": raw_centroids,
        }
        sizes_str = "  ".join(f"c{c}:{cluster_sizes.get(c, 0)}" for c in range(k))
        print(f"    {upos}: {len(idxs)} pairs, k={k}  [{sizes_str}]")

    out_tsv = lang_dir / f"profiles_b_k{n_clusters}.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("	".join(["lemma", "upos", "n", "cluster_id"] + feature_names) + "\n")
        for row_i, (lemma, upos, n) in enumerate(keys):
            prop_vals = "	".join(f"{v:.4f}" for v in X[row_i].tolist())
            f.write(f"{lemma}	{upos}	{n}	{labels[row_i]}	{prop_vals}\n")

    meta = {
        "lang": lang,
        "mode": "b",
        "n_clusters_requested": n_clusters,
        "min_count": min_count,
        "n_lemma_upos": len(keys),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_per_upos": dict(Counter(upos for _, upos, _ in keys)),
        "per_upos": per_upos_meta,
    }
    out_meta = lang_dir / f"profiles_b_k{n_clusters}_meta.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  -> {out_tsv}")
    print(f"  -> {out_meta}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_language(lang: str, output_dir: Path, args: argparse.Namespace) -> None:
    lang_dir = output_dir / lang
    obs = load_obs(lang_dir)
    if args.mode == "a":
        run_mode_a(lang, lang_dir, obs, args.min_count, args.threshold)
    else:
        run_mode_b(lang, lang_dir, obs, args.min_count, args.n_clusters)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dependency profiles (binarised or clustered) from observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", metavar="ISO",
                        help="Language ISO code.")
    parser.add_argument("--all-langs", action="store_true",
                        help="Process every lang with observations.pkl under --output-dir.")
    parser.add_argument("--output-dir", metavar="DIR", default="profiles",
                        help="Profiles root directory.")
    parser.add_argument("--mode", choices=["a", "b"], required=True,
                        help="a = binarise  |  b = k-means cluster")
    parser.add_argument("--min-count", type=int, default=20,
                        help="Min occurrences for a (lemma,upos) pair to receive a profile.")
    parser.add_argument("--threshold", type=float, default=0.33,
                        help="[mode a] Proportion threshold for binarising features.")
    parser.add_argument("--n-clusters", type=int, default=20,
                        help="[mode b] Number of k-means clusters.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all_langs:
        langs = sorted(
            p.name for p in output_dir.iterdir()
            if p.is_dir() and (p / "observations.pkl").exists()
        )
    elif args.lang:
        langs = [args.lang]
    else:
        parser.error("Provide --lang ISO or --all-langs.")

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  Language: {lang}  mode={args.mode}")
        print(f"{'='*60}")
        try:
            process_language(lang, output_dir, args)
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
