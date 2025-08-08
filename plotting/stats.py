import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import os


def parse_log_file(log_path):
    """Parses a log file to extract the 5-fold validation RMSE and MAE scores."""
    try:
        with open(log_path, "r") as f:
            content = f.read()
            # Regex to find the list of RMSE and MAE scores
            rmse_match = re.search(r"VAL FOLD RMSEs: \[(.*?)\]", content)
            mae_match = re.search(r"VAL FOLD MAEs: \[(.*?)\]", content)

            if rmse_match and mae_match:
                rmse_scores_str = rmse_match.group(1)
                mae_scores_str = mae_match.group(1)

                rmse_scores = [float(s.strip()) for s in rmse_scores_str.split(",")]
                mae_scores = [float(s.strip()) for s in mae_scores_str.split(",")]

                if len(rmse_scores) == 5 and len(mae_scores) == 5:
                    return rmse_scores, mae_scores
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_path}")
    except Exception as e:
        print(f"Warning: Could not parse log file {log_path}. Error: {e}")
    return None, None


def parse_csv_file(csv_path):
    """Parses the final results CSV to get all summary statistics."""
    try:
        df = pd.read_csv(csv_path)
        # Extract all relevant columns
        val_rmse_mean = df["val_rmse_mean"].iloc[0]
        val_rmse_std = df["val_rmse_std"].iloc[0]
        val_mae_mean = df["val_mae_mean"].iloc[0]
        val_mae_std = df["val_mae_std"].iloc[0]

        test_rmse_mean = df["test_rmse_mean"].iloc[0]
        test_rmse_ci_low = df["test_rmse_ci_low"].iloc[0]
        test_rmse_ci_high = df["test_rmse_ci_high"].iloc[0]

        test_mae_mean = df["test_mae_mean"].iloc[0]
        test_mae_ci_low = df["test_mae_ci_low"].iloc[0]
        test_mae_ci_high = df["test_mae_ci_high"].iloc[0]

        return {
            "val_rmse_mean": val_rmse_mean,
            "val_rmse_std": val_rmse_std,
            "val_mae_mean": val_mae_mean,
            "val_mae_std": val_mae_std,
            "test_rmse_mean": test_rmse_mean,
            "test_rmse_ci_low": test_rmse_ci_low,
            "test_rmse_ci_high": test_rmse_ci_high,
            "test_mae_mean": test_mae_mean,
            "test_mae_ci_low": test_mae_ci_low,
            "test_mae_ci_high": test_mae_ci_high,
        }
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Warning: Could not parse CSV file {csv_path}. Error: {e}")
    return None


def calculate_stats(main_scores, baseline_scores):
    """Calculates Wilcoxon p-value and Cohen's d for a given metric."""
    if main_scores is None or baseline_scores is None:
        return None, None

    try:
        _, p_value = wilcoxon(main_scores, baseline_scores, alternative="less")
    except ValueError:
        p_value = 1.0

    differences = np.array(main_scores) - np.array(baseline_scores)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = abs(mean_diff / std_diff) if std_diff != 0 else 0.0

    return p_value, cohens_d


def main(args):
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # --- FIX: Handle the main_model argument intelligently ---
    main_model_name_base = args.main_model
    if main_model_name_base == "polyatomic":
        main_model_id = "polyatomic_polyatomic"
    else:
        # This part assumes a format like "gcn_smiles" if you ever change the main model
        main_model_id = main_model_name_base

    output_dir.mkdir(exist_ok=True)

    # --- 1. Parse all experiment directories ---
    all_data = {}

    # --- FIX: Check if results_dir is a dataset dir or a parent of dataset dirs ---
    # Heuristic: if it contains a 'gcn' or 'gat' folder, it's a dataset directory.
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

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for log_file in model_dir.glob("*.txt"):
                base_name = log_file.stem
                clean_name = re.sub(r"_\d{8}_\d{6}$", "", base_name)

                # Handle the special case of 'polyatomic_polyatomic'
                if "polyatomic_polyatomic" in clean_name:
                    exp_id = "polyatomic_polyatomic"
                else:
                    exp_id = "_".join(clean_name.split("_")[:2])

                csv_file_name = re.sub(
                    r"_\d{8}_\d{6}$", "_final_results.csv", base_name
                )
                csv_file = log_file.with_name(csv_file_name)

                rmse_scores, mae_scores = parse_log_file(log_file)
                summary_stats = parse_csv_file(csv_file)

                if rmse_scores and mae_scores and summary_stats:
                    all_data[dataset_name][exp_id] = {
                        "rmse_folds": rmse_scores,
                        "mae_folds": mae_scores,
                        "summary": summary_stats,
                    }

    # --- 2. Perform statistical analysis for each dataset ---
    for dataset_name, experiments in all_data.items():
        if main_model_id not in experiments:
            print(
                f"Warning: Main model '{main_model_id}' not found for dataset '{dataset_name}'. Skipping stats."
            )
            continue

        main_model_rmse_folds = experiments[main_model_id]["rmse_folds"]
        baselines = {
            name: data for name, data in experiments.items() if name != main_model_id
        }

        p_values_uncorrected = []
        baseline_names_ordered = []

        for name, data in baselines.items():
            p_value, cohens_d = calculate_stats(
                main_model_rmse_folds, data["rmse_folds"]
            )
            data["p_value_raw"] = p_value
            data["cohens_d"] = cohens_d
            if p_value is not None:
                p_values_uncorrected.append(p_value)
                baseline_names_ordered.append(name)

        if p_values_uncorrected:
            _, p_values_corrected, _, _ = multipletests(
                p_values_uncorrected, alpha=0.05, method="holm"
            )
            for i, name in enumerate(baseline_names_ordered):
                baselines[name]["p_value_corrected"] = p_values_corrected[i]

    # --- 3. Generate the final report ---
    report_path = output_dir / f"{args.exp_name}_full_results_summary.txt"
    with open(report_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("Comprehensive Performance Summary\n")
        f.write("=" * 120 + "\n\n")

        for dataset_name in sorted(all_data.keys()):
            f.write(f"--- Dataset: {dataset_name.upper()} ---\n\n")
            header = (
                f"{'Model (Rep.)':<25} | {'Val RMSE':<20} | {'Val MAE':<20} | "
                f"{'Test RMSE':<25} | {'Test MAE':<25}\n"
            )
            f.write(header)
            f.write("-" * len(header) + "\n")

            experiments = all_data[dataset_name]
            for exp_id in sorted(experiments.keys()):
                summary = experiments[exp_id]["summary"]
                val_rmse_str = (
                    f"{summary['val_rmse_mean']:.3f} ± {summary['val_rmse_std']:.3f}"
                )
                val_mae_str = (
                    f"{summary['val_mae_mean']:.3f} ± {summary['val_mae_std']:.3f}"
                )
                test_rmse_str = f"{summary['test_rmse_mean']:.3f} [{summary['test_rmse_ci_low']:.3f}, {summary['test_rmse_ci_high']:.3f}]"
                test_mae_str = f"{summary['test_mae_mean']:.3f} [{summary['test_mae_ci_low']:.3f}, {summary['test_mae_ci_high']:.3f}]"

                f.write(
                    f"{exp_id:<25} | {val_rmse_str:<20} | {val_mae_str:<20} | "
                    f"{test_rmse_str:<25} | {test_mae_str:<25}\n"
                )

            f.write("\n--- Statistical Comparison (vs. " + main_model_id + ") ---\n\n")
            stat_header = f"{'Baseline Model':<25} | {'p-value (raw)':<15} | {'p-value (Holm)':<15} | {'Effect Size (d)':<15}\n"
            f.write(stat_header)
            f.write("-" * len(stat_header) + "\n")

            baselines = {k: v for k, v in experiments.items() if k != main_model_id}
            for exp_id in sorted(baselines.keys()):
                data = baselines[exp_id]
                p_raw = data.get("p_value_raw", "N/A")
                p_corr = data.get("p_value_corrected", "N/A")
                cohen_d = data.get("cohens_d", "N/A")

                p_raw_str = f"{p_raw:.4f}" if isinstance(p_raw, float) else p_raw
                p_corr_str = f"{p_corr:.4f}" if isinstance(p_corr, float) else p_corr
                cohen_d_str = (
                    f"{cohen_d:.2f}" if isinstance(cohen_d, float) else cohen_d
                )

                f.write(
                    f"{exp_id:<25} | {p_raw_str:<15} | {p_corr_str:<15} | {cohen_d_str:<15}\n"
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
        help="Name of the experiment/dataset.",
    )

    args = parser.parse_args()
    main(args)
