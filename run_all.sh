#!/bin/bash

# --- 1. Define your parameters here ---
# Special case for your model
PACTNET_MODEL="polyatomic"
PACTNET_REP="polyatomic"

# Baseline models and representations
BASELINE_MODELS=("gcn" "gin" "gat" "sage")
BASELINE_REPS=("smiles" "selfies" "ecfp")

# Datasets for all experiments
DATASETS=("esol" "freesolv" "lipophil" "boilingpoint" "qm9" "ic50" "bindingdb")

# --- 2. Generate all valid commands ---
#    We use a subshell `( ... )` to group the output of two separate loops.
#    This combined output is then piped to GNU Parallel.

(
  # Loop 1: Generate commands for your PACTNet model
  echo "--- Generating PACTNet Jobs ---" >&2
  for dataset in "${DATASETS[@]}"; do
    echo "python3 main.py --model $PACTNET_MODEL --rep $PACTNET_REP --dataset $dataset"
  done

  # Loop 2: Generate commands for all baseline combinations
  echo "--- Generating Baseline GNN Jobs ---" >&2
  for model in "${BASELINE_MODELS[@]}"; do
    for rep in "${BASELINE_REPS[@]}"; do
      for dataset in "${DATASETS[@]}"; do
        echo "python3 main.py --model $model --rep $rep --dataset $dataset"
      done
    done
  done
) | \
caffeinate -s parallel -j 10 --bar --eta

echo "All jobs complete."