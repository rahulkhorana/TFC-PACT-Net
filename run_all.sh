#!/bin/bash

# --- 1. Define your parameters here ---
MODELS=("gcn" "gin" "gat" "sage" "polyatomic")
REPS=("smiles" "selfies" "ecfp" "polyatomic")
DATASETS=("esol" "freesolv" "lipophil" "boilingpoint" "qm9" "ic50" "bindingdb")

# --- 2. Use GNU Parallel to run all combinations ---
#    This command keeps your Mac awake, runs 10 jobs in parallel,
#    shows a progress bar, and builds the commands for you.

caffeinate -s parallel -j 10 --bar --eta \
    'echo "Starting job: model={1}, rep={2}, dataset={3}"; python3 main.py --model {1} --rep {2} --dataset {3}' \
    ::: "${MODELS[@]}" \
    ::: "${REPS[@]}" \
    ::: "${DATASETS[@]}"

echo "All jobs complete."