#!/usr/bin/env bash
# Reruns the whole analysis from data/raw with one command (WORKSPACE.md rule 5).
#   bash workflows/run_all.sh
# Parameters are not accepted here on purpose: they live in configs/scrna.yaml.
set -euo pipefail

cd "$(dirname "$0")/.."

for step in \
    workflows/01_qc.py \
    workflows/02_normalize_cluster.py \
    workflows/03_annotate.py \
    workflows/04_composition.py \
    workflows/05_pseudobulk_de.py
do
    echo "=============== ${step} ==============="
    python "${step}"
done

echo "done — results/ and figures/ regenerated"
