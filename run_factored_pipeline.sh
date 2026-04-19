#!/usr/bin/env bash
# run_factored_pipeline.sh — Build everything needed for the factored LM.
#
# Languages: en + 14 non-English (silver: ar es eu fi hi hy id lv tr zh;
#            UD-only: de ja ru; ko via UD-Sejong only — Kaist excluded)
# Excluded: te (Telugu-MTG: zero FEATS/LEMMA), UD_Korean-Kaist (fused tokenization),
#           UD_Japanese-GSDLUW (LUW duplicate of GSD), UD_Arabic-NYUAD (FORM withheld)
#
# Steps:
#   1  Collect observations for 14 non-English languages
#   2  Compute K=8 clustered profiles for all 15 languages
#   3  Re-tokenize with lang codes + profiles -> output_factored/
#   4  Build factor vocabulary -> factor_vocab.json
#
# English observations + profiles already exist; steps 2-4 re-run or extend them.
#
# Usage:
#   bash run_factored_pipeline.sh
#   bash run_factored_pipeline.sh 2   # start from step 2

set -euo pipefail

PYTHON=".venv/Scripts/python"
UD_DIR="Universal Dependencies 2.17/ud-treebanks-v2.17/ud-treebanks-v2.17"
SILVER_DIR="silver_v2_output"
PROFILE_DIR="profiles"
OUTPUT_DIR="output_factored"
FACTOR_VOCAB="factor_vocab.json"
N_CLUSTERS=8
MIN_COUNT=20

START_STEP="${1:-1}"

echo "============================================================"
echo " Factored LM data pipeline"
echo " n_clusters=${N_CLUSTERS}  min_count=${MIN_COUNT}"
echo " Starting from step ${START_STEP}"
echo "============================================================"
echo ""

# ----------------------------------------------------------------
# Step 1: Collect observations for non-English languages
# ----------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
    echo "[1/4] Collecting observations for non-English languages..."
    for LANG in ar de es eu fi hi hy id ja ko lv ru tr zh; do
        echo ""
        echo "  --- ${LANG} ---"
        $PYTHON build_dep_profiles.py \
            --lang "$LANG" \
            --ud-dir "$UD_DIR" \
            --silver-dir "$SILVER_DIR" \
            --output-dir "$PROFILE_DIR"
    done
    echo ""
    echo "Step 1 done."
    echo ""
fi

# ----------------------------------------------------------------
# Step 2: Compute K=8 profiles for all 16 languages
# (en already has K=8 but re-running is harmless and keeps params consistent)
# ----------------------------------------------------------------
if [ "$START_STEP" -le 2 ]; then
    echo "[2/4] Computing K=${N_CLUSTERS} clustered profiles (mode b, all 15 languages)..."
    $PYTHON compute_profiles.py \
        --all-langs \
        --output-dir "$PROFILE_DIR" \
        --mode b \
        --n-clusters "$N_CLUSTERS" \
        --min-count "$MIN_COUNT"
    echo ""
    echo "Step 2 done."
    echo ""
fi

# ----------------------------------------------------------------
# Step 3: Tokenize with lang codes + K=8 profiles -> output_factored/
# ----------------------------------------------------------------
if [ "$START_STEP" -le 3 ]; then
    echo "[3/4] Tokenizing with lang codes and dependency profiles..."
    $PYTHON delexicalize.py \
        --ud-dir "$UD_DIR" \
        --silver-dir "$SILVER_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --include-lang-code \
        --langs ar,de,en,es,eu,fi,hi,hy,id,ja,ko,lv,ru,tr,zh \
        --max-tokens-per-lang 2000000 \
        --profile-dir "$PROFILE_DIR" \
        --profile-mode b \
        --profile-n-clusters "$N_CLUSTERS"
    echo ""
    echo "Step 3 done."
    echo ""
fi

# ----------------------------------------------------------------
# Step 4: Build factor vocabulary
# ----------------------------------------------------------------
if [ "$START_STEP" -le 4 ]; then
    echo "[4/4] Building factor vocabulary..."
    $PYTHON build_factor_vocab.py \
        --vocab "${OUTPUT_DIR}/delex_vocab.txt" \
        --output "$FACTOR_VOCAB" \
        --min-freq 3
    echo ""
    echo "Step 4 done."
    echo ""
fi

echo "============================================================"
echo " Pipeline complete."
echo "   Tokenized data : ${OUTPUT_DIR}/"
echo "   Factor vocab   : ${FACTOR_VOCAB}"
echo ""
echo " Next: submit hyperparameter search to cluster"
echo "   bash cluster/submit_hparam.sh 1"
echo "============================================================"
