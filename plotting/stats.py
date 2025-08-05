import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def parse_log_file(log_path):
    """Parses a log file to extract the 5-fold validation RMSE scores."""
    try:
        with open(log_path, "r") as f:
            content = f.read()
            # Regex to find the list of RMSE scores
            match = re.search(r"VAL FOLD RMSEs: \[(.*?)\]", content)
            if match:
                # Extract the string of numbers, split by comma, and convert to float
                scores_str = match.group(1)
                scores = [float(s.strip()) for s in scores_str.split(",")]
                if len(scores) == 5:
                    return scores
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_path}")
    except Exception as e:
        print(f"Warning: Could not parse log file {log_path}. Error: {e}")
    return None


def parse_csv_file(csv_path):
    """Parses the final results CSV to get mean and std dev of validation RMSE."""
    try:
        df = pd.read_csv(csv_path)
        # Assumes columns are named 'val_rmse_mean' and 'val_rmse_std'
        mean_rmse = df["val_rmse_mean"].iloc[0]
        std_rmse = df["val_rmse_std"].iloc[0]
        return mean_rmse, std_rmse
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Warning: Could not parse CSV file {csv_path}. Error: {e}")
    return None, None


def calculate_stats(main_scores, baseline_scores):
    """Calculates Wilcoxon p-value and Cohen's d."""
    if main_scores is None or baseline_scores is None:
        return None, None

    # Wilcoxon signed-rank test (one-sided)
    # We test if the main model's scores are significantly 'less' than the baseline's
    try:
        _, p_value = wilcoxon(main_scores, baseline_scores, alternative="less")
    except ValueError:
        # This can happen if all differences are zero
        p_value = 1.0

    # Cohen's d for paired samples
    differences = np.array(main_scores) - np.array(baseline_scores)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = abs(mean_diff / std_diff) if std_diff != 0 else 0.0

    return p_value, cohens_d


def main(args):
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    main_model_name = args.main_model

    output_dir.mkdir(exist_ok=True)

    model_data = {}

    # --- 1. Parse all model directories ---
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name

            # Find the log and csv files within the model directory
            try:
                log_file = next(model_dir.glob("*.txt"))
                csv_file = next(model_dir.glob("*.csv"))
            except StopIteration:
                print(f"Warning: Skipping {model_name}, couldn't find log or csv file.")
                continue

            fold_scores = parse_log_file(log_file)
            mean_rmse, std_rmse = parse_csv_file(csv_file)

            if fold_scores and mean_rmse is not None:
                model_data[model_name] = {
                    "fold_scores": fold_scores,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                }

    if main_model_name not in model_data:
        raise ValueError(
            f"Main model '{main_model_name}' not found in results directory."
        )

    main_model_scores = model_data[main_model_name]["fold_scores"]

    # --- 2. Calculate stats for each baseline ---
    baselines = {
        name: data for name, data in model_data.items() if name != main_model_name
    }
    p_values_uncorrected = []
    baseline_names_ordered = []

    for name, data in baselines.items():
        p_value, cohens_d = calculate_stats(main_model_scores, data["fold_scores"])
        data["p_value_raw"] = p_value
        data["cohens_d"] = cohens_d
        if p_value is not None:
            p_values_uncorrected.append(p_value)
            baseline_names_ordered.append(name)

    # --- 3. Apply Holm-Bonferroni correction ---
    if p_values_uncorrected:
        reject, p_values_corrected, _, _ = multipletests(
            p_values_uncorrected, alpha=0.05, method="holm"
        )
        for i, name in enumerate(baseline_names_ordered):
            baselines[name]["p_value_corrected"] = p_values_corrected[i]

    # --- 4. Generate the report ---
    report_path = output_dir / "statistical_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("Comparative Performance Analysis\n")
        f.write("=" * 80 + "\n")
        header = f"{'Model':<25} | {'Mean RMSE (± Std. Dev.)':<28} | {'p-value':>10} | {'Corrected p-value':>18} | {'Effect Size (d)':>18}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        # Write baseline results
        for name, data in sorted(baselines.items()):
            mean_std_str = f"{data['mean_rmse']:.4f} (± {data['std_rmse']:.4f})"
            p_raw_str = (
                f"{data.get('p_value_raw', 'N/A'):.4f}"
                if data.get("p_value_raw") is not None
                else "N/A"
            )
            p_corr_str = (
                f"{data.get('p_value_corrected', 'N/A'):.4f}"
                if data.get("p_value_corrected") is not None
                else "N/A"
            )
            d_str = (
                f"{data.get('cohens_d', 'N/A'):.2f}"
                if data.get("cohens_d") is not None
                else "N/A"
            )
            f.write(
                f"{name:<25} | {mean_std_str:<28} | {p_raw_str:>10} | {p_corr_str:>18} | {d_str:>18}\n"
            )

        # Write main model result
        main_data = model_data[main_model_name]
        mean_std_str = f"{main_data['mean_rmse']:.4f} (± {main_data['std_rmse']:.4f})"
        f.write("-" * len(header) + "\n")
        f.write(
            f"{main_model_name + ' (Main)':<25} | {mean_std_str:<28} | {'-':>10} | {'-':>18} | {'-':>18}\n"
        )

    print(f"Report successfully generated at: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collate and analyze model performance results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing subdirectories for each model's results.",
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
        help="Name of the main model directory to compare others against.",
    )

    args = parser.parse_args()
    main(args)
