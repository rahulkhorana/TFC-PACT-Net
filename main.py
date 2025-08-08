import argparse

# The runners module will contain the high-level logic for each model type
from runners import run_gnn_experiment, run_gp_experiment


def main():
    """
    Main entry point for running all experiments.
    Parses command-line arguments and calls the appropriate runner function.
    """
    parser = argparse.ArgumentParser(
        description="Run GNN or GP experiments for molecular property prediction."
    )
    parser.add_argument(
        "--model",
        choices=["gcn", "gin", "gat", "sage", "gp", "polyatomic"],
        required=True,
        help="The model to train and evaluate.",
    )
    parser.add_argument(
        "--rep",
        choices=["smiles", "selfies", "ecfp", "polyatomic"],
        required=True,
        help="The molecular representation to use.",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "esol",
            "freesolv",
            "lipophil",
            "boilingpoint",
            "qm9",
            "ic50",
            "bindingdb",
        ],
        required=True,
        help="The dataset to use for the experiment.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run for hyperparameter search in each fold.",
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.model == "gp" and args.rep == "polyatomic":
        raise ValueError(
            "The 'polyatomic' representation is not compatible with the 'gp' model."
        )
    if args.model == "polyatomic" and args.rep != "polyatomic":
        raise ValueError(
            "The 'polyatomic' model must be used with the 'polyatomic' representation."
        )

    # --- Delegate to the correct runner ---
    if args.model == "gp":
        run_gp_experiment(args)
    else:
        run_gnn_experiment(args)


if __name__ == "__main__":
    main()
