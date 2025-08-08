import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from math import sqrt
from tabulate import tabulate

np.random.seed(42)


def parse_log_file(log_path):
    """
    Parses a log file to extract the 5-fold validation RMSE and MAE scores.
    This function reads the text file and uses regular expressions to find the
    lines containing the validation scores.
    """
    try:
        with open(log_path, "r") as f:
            content = f.read()
            # Regex to find the list of RMSE and MAE scores for validation folds
            rmse_match = re.search(r"VAL FOLD RMSEs: \[(.*?)\]", content)
            mae_match = re.search(r"VAL FOLD MAEs: \[(.*?)\]", content)

            if rmse_match and mae_match:
                # Extract the string of comma-separated numbers
                rmse_scores_str = rmse_match.group(1)
                mae_scores_str = mae_match.group(1)

                # Convert the string of scores into a list of floats
                rmse_scores = [float(s.strip()) for s in rmse_scores_str.split(",")]
                mae_scores = [float(s.strip()) for s in mae_scores_str.split(",")]

                # Ensure we have exactly 5 scores for a complete 5-fold validation
                if len(rmse_scores) == 5 and len(mae_scores) == 5:
                    return rmse_scores, mae_scores
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_path}")
    except Exception as e:
        print(f"Warning: Could not parse log file {log_path}. Error: {e}")
    # Return None if parsing fails or file is incomplete
    return None, None


def parse_csv_file(csv_path):
    """
    Parses the final results CSV to get all summary statistics, including
    means, standard deviations, and confidence intervals.
    """
    try:
        df = pd.read_csv(csv_path)
        # Extract all relevant columns from the CSV file
        return {
            "val_rmse_mean": df["val_rmse_mean"].iloc[0],
            "val_rmse_std": df["val_rmse_std"].iloc[0],
            "val_mae_mean": df["val_mae_mean"].iloc[0],
            "val_mae_std": df["val_mae_std"].iloc[0],
            "test_rmse_mean": df["test_rmse_mean"].iloc[0],
            "test_rmse_ci_low": df["test_rmse_ci_low"].iloc[0],
            "test_rmse_ci_high": df["test_rmse_ci_high"].iloc[0],
            "test_mae_mean": df["test_mae_mean"].iloc[0],
            "test_mae_ci_low": df["test_mae_ci_low"].iloc[0],
            "test_mae_ci_high": df["test_mae_ci_high"].iloc[0],
        }
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Warning: Could not parse CSV file {csv_path}. Error: {e}")
    return None


def main(args):
    """
    Main function to drive the parsing, statistical analysis, and report generation.
    """
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- 1. Parse all experiment directories ---
    all_data = {}
    # Determine if the results_dir is a single dataset or a parent of multiple datasets
    is_single_dataset = any(
        p.name in ["gcn", "gat", "gin", "sage", "polyatomic"]
        for p in results_dir.iterdir()
    )
    dataset_dirs = (
        [results_dir]
        if is_single_dataset
        else [p for p in results_dir.iterdir() if p.is_dir()]
    )

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        all_data[dataset_name] = {}

        # Iterate through each model family directory (e.g., gcn, gat, polyatomic)
        for model_family_dir in dataset_dir.iterdir():
            if not model_family_dir.is_dir():
                continue

            # *** THE FIX IS HERE ***
            # Use .rglob("*.txt") to recursively search for log files.
            # This finds files in both flat structures (like 'polyatomic') and
            # nested structures (like 'gcn/gcn_ecfp_qm9/').
            for log_file in model_family_dir.rglob("*.txt"):
                base_name = log_file.stem
                # Clean the timestamp from the filename to get a consistent name
                clean_name = re.sub(r"_\d{8}_\d{6}$", "", base_name)

                # Standardize the experiment ID for consistent grouping
                if args.main_model in clean_name:
                    exp_id = args.main_model
                else:
                    # For GNNs, create an ID like 'gcn_ecfp'
                    exp_id = "_".join(clean_name.split("_")[:2])

                # Construct the corresponding CSV filename
                csv_file_name = re.sub(
                    r"_\d{8}_\d{6}$", "_final_results.csv", base_name
                )
                csv_file = log_file.with_name(csv_file_name)

                # Parse both the log and CSV files
                rmse_scores, mae_scores = parse_log_file(log_file)
                summary_stats = parse_csv_file(csv_file)

                # If all data is successfully parsed, add it to our main dictionary
                if rmse_scores and mae_scores and summary_stats:
                    if exp_id not in all_data[dataset_name]:
                        all_data[dataset_name][exp_id] = {
                            "rmse_folds": rmse_scores,
                            "mae_folds": mae_scores,
                            "summary": summary_stats,
                        }

    # --- 2. Perform statistical analysis for each dataset ---
    for dataset_name, experiments in all_data.items():
        main_model_id = args.main_model
        if main_model_id not in experiments:
            print(
                f"Warning: Main model '{main_model_id}' not found for dataset '{dataset_name}'. Skipping stats."
            )
            continue

        model_names = sorted(experiments.keys())
        # Prepare data for Friedman test (rows=folds, columns=models)
        rmse_scores_by_model = [experiments[name]["rmse_folds"] for name in model_names]

        if len(model_names) < 2 or not all(len(s) == 5 for s in rmse_scores_by_model):
            continue

        friedman_data = np.array(rmse_scores_by_model).T

        try:
            # The '*' unpacks the columns for the function
            friedman_statistic, friedman_p_value = friedmanchisquare(*friedman_data.T)
        except ValueError:
            friedman_statistic, friedman_p_value = np.nan, 1.0

        friedman_results = {
            "statistic": friedman_statistic,
            "p_value": friedman_p_value,
            "post_hoc": {},
        }

        # If Friedman test is significant, proceed to Nemenyi post-hoc test
        if friedman_p_value < 0.05:
            k = len(model_names)
            n = 5  # Number of folds

            # Rank models for each fold. Lower RMSE gets a lower rank.
            ranks = np.argsort(np.argsort(friedman_data, axis=1)) + 1
            mean_ranks = np.mean(ranks, axis=0)

            # Critical Difference (CD) calculation for Nemenyi test
            # Using pre-computed q-values for alpha=0.05
            q_alpha = {
                2: 1.96,
                3: 2.34,
                4: 2.57,
                5: 2.77,
                6: 2.94,
                7: 3.07,
                8: 3.19,
                9: 3.29,
                10: 3.38,
                11: 3.46,
                12: 3.53,
                13: 3.59,
                14: 3.65,
                15: 3.70,
            }.get(
                k, 3.9
            )  # Default for k > 15
            CD = q_alpha * sqrt(k * (k + 1) / (6 * n))

            main_model_idx = model_names.index(main_model_id)
            main_model_rank = mean_ranks[main_model_idx]

            # Compare the main model against all other models
            for i in range(k):
                if i == main_model_idx:
                    continue
                diff = abs(main_model_rank - mean_ranks[i])
                is_significant = diff > CD
                baseline_name = model_names[i]
                friedman_results["post_hoc"][f"{main_model_id} vs {baseline_name}"] = {
                    "mean_rank_diff": diff,
                    "critical_diff": CD,
                    "significant": is_significant,
                }
        experiments["statistical_results"] = friedman_results

    # --- 3. Generate the final report ---
    report_path = output_dir / f"{args.exp_name}_full_results_summary.txt"
    with open(report_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("Comprehensive Performance Summary\n")
        f.write("=" * 120 + "\n\n")

        for dataset_name in sorted(all_data.keys()):
            f.write(f"--- Dataset: {dataset_name.upper()} ---\n\n")
            header = (
                f"{'Model':<25} | {'Val RMSE':<20} | {'Val MAE':<20} | "
                f"{'Test RMSE':<25} | {'Test MAE':<25}\n"
            )
            f.write(header)
            f.write("-" * (len(header) + 5) + "\n")

            experiments = all_data[dataset_name]
            model_keys = sorted(
                [k for k in experiments.keys() if k != "statistical_results"]
            )

            for exp_id in model_keys:
                summary = experiments[exp_id]["summary"]
                val_rmse_str = (
                    f"{summary['val_rmse_mean']:.4f} ± {summary['val_rmse_std']:.4f}"
                )
                val_mae_str = (
                    f"{summary['val_mae_mean']:.4f} ± {summary['val_mae_std']:.4f}"
                )
                test_rmse_str = f"{summary['test_rmse_mean']:.4f} [{summary['test_rmse_ci_low']:.4f}, {summary['test_rmse_ci_high']:.4f}]"
                test_mae_str = f"{summary['test_mae_mean']:.4f} [{summary['test_mae_ci_low']:.4f}, {summary['test_mae_ci_high']:.4f}]"

                f.write(
                    f"{exp_id:<25} | {val_rmse_str:<20} | {val_mae_str:<20} | "
                    f"{test_rmse_str:<25} | {test_mae_str:<25}\n"
                )

            friedman_res = experiments.get("statistical_results")
            if friedman_res and friedman_res["post_hoc"]:
                f.write(
                    "\n--- Statistical Analysis (Friedman + Nemenyi Post-Hoc) ---\n"
                )
                f.write(
                    f"Friedman Test: Chi-Square = {friedman_res['statistic']:.4f}, p-value = {friedman_res['p_value']:.4f}\n"
                )
                if friedman_res["p_value"] < 0.05:
                    f.write(
                        "-> The overall performance of the models is statistically different.\n"
                    )
                    cd = list(friedman_res["post_hoc"].values())[0]["critical_diff"]
                    f.write(
                        f"\n--- Nemenyi Post-Hoc Test (Critical Difference = {cd:.4f}) ---\n"
                    )

                    table_headers = ["Comparison", "Mean Rank Diff", "Result"]
                    table_data = []

                    for comp_key, post_hoc_data in sorted(
                        friedman_res["post_hoc"].items()
                    ):
                        result = (
                            "Significant"
                            if post_hoc_data["significant"]
                            else "Not Significant"
                        )
                        table_data.append(
                            [comp_key, f"{post_hoc_data['mean_rank_diff']:.4f}", result]
                        )

                    f.write(
                        tabulate(table_data, headers=table_headers, tablefmt="plain")
                        + "\n"
                    )
                else:
                    f.write(
                        "-> No statistically significant difference found between models.\n"
                    )

            f.write("\n" + "=" * 120 + "\n\n")

    print(f"Report successfully generated at: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collate and analyze model performance results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing results. Can be a parent of dataset dirs, or a single dataset dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the final report will be saved.",
    )
    parser.add_argument(
        "--main_model",
        type=str,
        required=True,
        help="Base name of the main model to compare against (e.g., 'polyatomic').",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Name of the experiment/dataset for the output file.",
    )

    args = parser.parse_args()
    main(args)
