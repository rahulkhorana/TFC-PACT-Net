import numpy as np
import pandas as pd
from pathlib import Path

# Import your project's modules
from data.data_handling import prepare_and_load_data, GP_FEATURIZERS
from training.training_pipeline import k_fold_tuned_eval
from models.gp import TanimotoGP as GPModel
from data.loaders import load_dataset

from training.train_eval import (
    k_fold_eval as gp_k_fold_eval,
    train_gp_model,
    eval_gp_model,
)


def run_gnn_experiment(args):
    """
    Orchestrates a full, tuned GNN experiment using the EFFICIENT pipeline.
    """
    print(
        f"--- Starting GNN Experiment: {args.model} | {args.rep} | {args.dataset} ---"
    )

    # 1. Prepare data ONCE. This function handles caching.
    train_graphs, test_graphs = prepare_and_load_data(args)

    # 2. Run the rigorous evaluation pipeline with the pre-featurized data.
    results_df = k_fold_tuned_eval(args, train_graphs, test_graphs)

    # 3. Save detailed results to a CSV file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_filename = f"{args.model}_{args.rep}_{args.dataset}_results.csv"
    results_df.to_csv(results_dir / results_filename, index=False)

    print(
        f"--- GNN Experiment Finished. Results saved to {results_dir / results_filename} ---"
    )


def run_gp_experiment(args):
    """Orchestrates a GP experiment, preserving the original logic."""
    print(f"--- Starting GP Experiment: {args.rep} | {args.dataset} ---")

    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    gp_feat = GP_FEATURIZERS[args.rep]

    X_train_feat = np.stack([gp_feat(x) for x in X_train]).astype(np.float32)
    X_test_feat = np.stack([gp_feat(x) for x in X_test]).astype(np.float32)

    def train_fn(X_tr, y_tr, log_file):
        model = GPModel()
        return train_gp_model(model, X_tr, y_tr, log_file)

    def eval_fn(model, X_te, y_te, log_file, scaler, return_preds=False):
        return eval_gp_model(
            model, X_te, y_te, log_file, scaler, return_preds=return_preds
        )

    _, test_metrics = gp_k_fold_eval(
        train_fn=train_fn,
        eval_fn=eval_fn,
        X_train=X_train_feat,
        y_train=y_train,
        model_name="gp",
        rep_name=args.rep,
        dataset_name=args.dataset,
        X_test=X_test_feat,
        y_test=y_test,
    )
    print(f"--- GP Experiment Finished. Final Test Metrics: {test_metrics} ---")
