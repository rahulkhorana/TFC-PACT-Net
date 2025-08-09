import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from tabulate import tabulate
import scikit_posthocs as sp


np.random.seed(42)


def parse_log_file(log_path):
    """
    Parses a log file to extract the 5-fold validation RMSE and MAE scores.
    """
    try:
        with open(log_path, "r") as f:
            content = f.read()
            rmse_match = re.search(r"VAL FOLD RMSEs: \[(.*?)\]", content)
            mae_match = re.search(r"VAL FOLD MAEs: \[(.*?)\]", content)
            if rmse_match and mae_match:
                rmse_scores = [float(s.strip()) for s in rmse_match.group(1).split(",")]
                mae_scores = [float(s.strip()) for s in mae_match.group(1).split(",")]
                if len(rmse_scores) == 5 and len(mae_scores) == 5:
                    return rmse_scores, mae_scores
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_path}")
    except Exception as e:
        print(f"Warning: Could not parse log file {log_path}. Error: {e}")
    return None, None


def parse_csv_file(csv_path):
    """
    Parses the final results CSV to get all summary statistics.
    """
    try:
        df = pd.read_csv(csv_path)
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


def interpret_kendalls_w(w):
    """Provides a qualitative interpretation of Kendall's W effect size."""
    if w < 0.1:
        return "Negligible effect"
    if w < 0.3:
        return "Small effect"
    if w < 0.5:
        return "Moderate effect"
    return "Large effect"


def main(args):
    """
    Main function to drive the parsing, statistical analysis, and report generation.
    """
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- 1. Parse all experiment directories ---
    all_data = {}
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

        for model_family_dir in dataset_dir.iterdir():
            if not model_family_dir.is_dir():
                continue

            # CORRECTED: Use .rglob() to recursively find log files in any structure.
            for log_file in model_family_dir.rglob("*.txt"):
                base_name = log_file.stem
                clean_name = re.sub(r"_\d{8}_\d{6}$", "", base_name)
                exp_id = (
                    args.main_model
                    if args.main_model in clean_name
                    else "_".join(clean_name.split("_")[:2])
                )

                csv_file_name = re.sub(
                    r"_\d{8}_\d{6}$", "_final_results.csv", base_name
                )
                csv_file = log_file.with_name(csv_file_name)

                rmse_scores, mae_scores = parse_log_file(log_file)
                summary_stats = parse_csv_file(csv_file)

                if rmse_scores and mae_scores and summary_stats:
                    if exp_id not in all_data[dataset_name]:
                        all_data[dataset_name][exp_id] = {
                            "rmse_folds": rmse_scores,
                            "summary": summary_stats,
                        }

    # --- 2. Perform statistical analysis for each dataset ---
    for dataset_name, experiments in all_data.items():
        if len(experiments) < 2:
            continue

        model_names = sorted(experiments.keys())
        rmse_data = np.array([experiments[name]["rmse_folds"] for name in model_names])

        # Run the Friedman test
        try:
            friedman_stat, p_value = friedmanchisquare(*rmse_data)
        except ValueError:
            friedman_stat, p_value = np.nan, 1.0

        # NEW: Calculate Kendall's W effect size
        k = len(model_names)  # number of models
        n = len(rmse_data[0])  # number of folds
        kendalls_w = friedman_stat / (n * (k - 1)) if n * (k - 1) != 0 else 0

        experiments["statistical_results"] = {
            "friedman_stat": friedman_stat,
            "p_value": p_value,
            "kendalls_w": kendalls_w,
            "post_hoc": None,
        }

        # NEW: Replace Nemenyi with Holm's post-hoc test if Friedman is significant
        if p_value < 0.05 and args.main_model in model_names:
            # Reshape data for scikit-posthocs (rows are folds, columns are models)
            df_posthoc = pd.DataFrame(rmse_data.T, columns=model_names)
            # Perform Holm's test against the specified control model
            holm_results = sp.posthoc_conover_friedman(df_posthoc, p_adjust="holm")
            experiments["statistical_results"]["post_hoc"] = holm_results

    # --- 3. Generate the final, enhanced report ---
    report_path = output_dir / f"{args.exp_name}_full_results_summary.txt"
    with open(report_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("Comprehensive Performance and Statistical Analysis Summary\n")
        f.write("=" * 100 + "\n\n")

        for dataset_name in sorted(all_data.keys()):
            f.write(f"--- Dataset: {dataset_name.upper()} ---\n\n")
            header = f"{'Model':<25} | {'Test RMSE (95% CI)':<30} | {'Validation RMSE (Mean ± StDev)':<35}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            experiments = all_data[dataset_name]
            model_keys = sorted(
                [k for k in experiments.keys() if k != "statistical_results"]
            )

            for exp_id in model_keys:
                summary = experiments[exp_id]["summary"]
                val_rmse_str = (
                    f"{summary['val_rmse_mean']:.4f} ± {summary['val_rmse_std']:.4f}"
                )
                test_rmse_str = f"{summary['test_rmse_mean']:.4f} [{summary['test_rmse_ci_low']:.4f}, {summary['test_rmse_ci_high']:.4f}]"
                f.write(f"{exp_id:<25} | {test_rmse_str:<30} | {val_rmse_str:<35}\n")

            stats_res = experiments.get("statistical_results")
            if stats_res:
                f.write("\n--- Statistical Analysis ---\n")
                f.write(
                    f"Friedman Test: Chi-Square = {stats_res['friedman_stat']:.4f}, p-value = {stats_res['p_value']:.4f}\n"
                )

                # Report Kendall's W
                w_interp = interpret_kendalls_w(stats_res["kendalls_w"])
                f.write(
                    f"Kendall's W Effect Size: {stats_res['kendalls_w']:.4f} ({w_interp})\n"
                )

                if stats_res["p_value"] < 0.05:
                    f.write(
                        "-> The overall performance of the models is statistically different.\n"
                    )
                    if stats_res["post_hoc"] is not None:
                        f.write(
                            f"\n--- Holm's Post-Hoc Test (Control: {args.main_model}) ---\n"
                        )
                        # We only need the p-values column against the control
                        holm_df = stats_res["post_hoc"][[args.main_model]].rename(
                            columns={args.main_model: "Adjusted p-value"}
                        )
                        holm_df["Significant"] = holm_df["Adjusted p-value"] < 0.05
                        f.write(tabulate(holm_df, headers="keys", tablefmt="plain"))
                        f.write("\n")
                else:
                    f.write(
                        "-> No statistically significant difference found between models.\n"
                    )

            f.write("\n" + "=" * 100 + "\n\n")

    print(f"Report successfully generated at: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collate and analyze model performance results."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory containing results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the final report.",
    )
    parser.add_argument(
        "--main_model",
        type=str,
        required=True,
        help="Base name of the control model (e.g., 'polyatomic').",
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name for the output report file."
    )
    args = parser.parse_args()
    main(args)
