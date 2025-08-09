"""
Single-dataset statistical comparison: control vs competitors via
Nadeau–Bengio corrected t-tests on outer folds (k-fold CV), with Holm FWER control.
No Wilcoxon. No Friedman. Test metrics are shown for context only.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from math import sqrt

from statsmodels.stats.multitest import multipletests  # Holm
from scipy.stats import t as tdist  # t distribution for NB p-values

np.random.seed(42)

# -----------------------------
# parse helpers
# -----------------------------


def _parse_list_in_brackets(inner: str) -> Optional[List[float]]:
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            return None
    return out


def parse_log_file(
    log_path: Path, expected_k: int = 5
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Extract K outer-fold validation RMSE and MAE arrays.
    Looks for:
      VAL FOLD RMSEs: [v1, ..., vK]
      VAL FOLD MAEs:  [v1, ..., vK]
    """
    try:
        text = log_path.read_text()
    except Exception:
        return None, None

    rmse_match = re.search(
        r"VAL\s+FOLD\s+RMSEs:\s*\[(.*?)\]", text, flags=re.IGNORECASE
    )
    mae_match = re.search(r"VAL\s+FOLD\s+MAEs:\s*\[(.*?)\]", text, flags=re.IGNORECASE)
    if not (rmse_match and mae_match):
        return None, None

    rmse_scores = _parse_list_in_brackets(rmse_match.group(1))
    mae_scores = _parse_list_in_brackets(mae_match.group(1))
    if rmse_scores is None or mae_scores is None:
        return None, None
    if len(rmse_scores) != expected_k or len(mae_scores) != expected_k:
        return None, None
    return rmse_scores, mae_scores


def parse_csv_file(csv_path: Path) -> Optional[Dict[str, float]]:
    """
    Parse *_final_results.csv with expected columns:
      val_rmse_mean, val_rmse_std, val_mae_mean, val_mae_std,
      test_rmse_mean, test_rmse_ci_low, test_rmse_ci_high,
      test_mae_mean,  test_mae_ci_low,  test_mae_ci_high
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    cols = [
        "val_rmse_mean",
        "val_rmse_std",
        "val_mae_mean",
        "val_mae_std",
        "test_rmse_mean",
        "test_rmse_ci_low",
        "test_rmse_ci_high",
        "test_mae_mean",
        "test_mae_ci_low",
        "test_mae_ci_high",
    ]
    if not all(c in df.columns for c in cols):
        return None
    row = df.iloc[0]
    return {k: float(row[k]) for k in cols}


def clean_base_and_exp_id(base_name: str) -> Tuple[str, str]:
    """
    Strip trailing _YYYYMMDD_HHMMSS. exp_id := first two tokens (e.g., 'gat_ecfp'),
    or 'polyatomic_polyatomic'.
    """
    clean = re.sub(r"_\d{8}_\d{6}$", "", base_name)
    toks = clean.split("_")
    exp_id = "_".join(toks[:2]) if len(toks) >= 2 else clean
    return clean, exp_id


def discover_dataset(results_dir: Path, expected_k: int = 5) -> Dict[str, Dict]:
    """
    Scan one dataset dir. Keep first valid (log,csv) per exp_id.
    Returns:
      data[exp_id] = {
        'rmse_folds': [K] or None,
        'mae_folds' : [K] or None,
        'summary'   : {...},
        'log_path'  : str,
        'csv_path'  : str
      }
    """
    data: Dict[str, Dict] = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for log_file in sorted(model_dir.glob("*.txt")):
            base = log_file.stem
            clean, exp_id = clean_base_and_exp_id(base)
            csv_file = log_file.with_name(f"{clean}_final_results.csv")
            if not csv_file.exists():
                continue
            rmse_scores, mae_scores = parse_log_file(log_file, expected_k=expected_k)
            summary_stats = parse_csv_file(csv_file)
            if summary_stats is None:
                continue
            if exp_id not in data:
                data[exp_id] = {
                    "rmse_folds": rmse_scores,
                    "mae_folds": mae_scores,
                    "summary": summary_stats,
                    "log_path": str(log_file),
                    "csv_path": str(csv_file),
                }
    return data


# -----------------------------
# Nadeau–Bengio corrected t (within dataset)
# -----------------------------


def nb_corrected_t(diffs: np.ndarray, k: int) -> Tuple[float, float, float]:
    """
    Nadeau–Bengio corrected resampled t for k-fold CV differences.
    diffs: array length k with (competitor - control) per fold (loss metric).
           Negative mean favors control. Returns (t_stat, SE_NB, mean_diff).
    Correction uses rho0 = 1/(k-1) → SE_NB = sqrt((1/k + 1/(k-1)) * s^2),
    where s^2 is unbiased sample variance across folds.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size != k:
        raise ValueError("diffs length must equal k")
    mean_d = float(diffs.mean())
    if k <= 1:
        return np.nan, np.nan, mean_d
    s2 = float(np.var(diffs, ddof=1))
    rho0 = 1.0 / (k - 1.0)
    se_nb = sqrt((1.0 / k + rho0) * s2)
    t_stat = (
        mean_d / se_nb
        if se_nb > 0
        else (-np.inf if mean_d < 0 else np.inf if mean_d > 0 else 0.0)
    )
    return t_stat, se_nb, mean_d


# -----------------------------
# main
# -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Single-dataset comparison: NB-corrected t on outer folds (control vs all) with Holm; prints Test metrics for context."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Dataset directory (e.g., ./logs_hyperparameter/qm9) with model-family subfolders.",
    )
    ap.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the report."
    )
    ap.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Base name for the output report file.",
    )
    ap.add_argument(
        "--control_model",
        type=str,
        required=True,
        help="Control exp_id (e.g., 'polyatomic_polyatomic'). Unique substring allowed.",
    )
    ap.add_argument(
        "--k", type=int, default=5, help="Expected number of outer folds (default: 5)."
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FWER level for Holm; also used for NB CIs if --print_ci.",
    )
    ap.add_argument(
        "--print_ci",
        action="store_true",
        help="Also print NB-style CIs for the mean fold difference.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{args.exp_name}_NB_single_dataset.txt"

    data = discover_dataset(results_dir, expected_k=args.k)
    if not data:
        raise SystemExit(
            f"No models found under {results_dir} with valid logs + *_final_results.csv"
        )

    exp_ids = sorted(data.keys())

    # match control (exact or unique substring)
    ctrl = args.control_model
    if ctrl not in data:
        matches = [e for e in exp_ids if ctrl.lower() in e.lower()]
        if len(matches) == 1:
            ctrl = matches[0]
        else:
            raise SystemExit(
                f"Control '{args.control_model}' not found. Available exp_ids: {exp_ids}"
            )

    with open(report_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write(
            f"Dataset: {results_dir.name} — Control vs competitors (NB-corrected t on outer folds; Holm across competitors)\n"
        )
        f.write("=" * 110 + "\n\n")
        f.write(f"Control exp_id: {ctrl}\n")
        f.write(f"k folds: {args.k}, alpha: {args.alpha}\n\n")

        # context table
        header = f"{'Model (exp_id)':<26} | {'Test RMSE (95% CI)':<30} | {'Test MAE (95% CI)':<30} | {'Val RMSE mean±sd':<22} | {'Val MAE mean±sd':<22}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for exp_id in exp_ids:
            s = data[exp_id]["summary"]
            line = (
                f"{exp_id:<26} | "
                f"{s['test_rmse_mean']:.6f} [{s['test_rmse_ci_low']:.6f}, {s['test_rmse_ci_high']:.6f}]  | "
                f"{s['test_mae_mean']:.6f}  [{s['test_mae_ci_low']:.6f},  {s['test_mae_ci_high']:.6f}]  | "
                f"{s['val_rmse_mean']:.6f} ± {s['val_rmse_std']:.6f}    | "
                f"{s['val_mae_mean']:.6f} ± {s['val_mae_std']:.6f}\n"
            )
            f.write(line)
        f.write("\n")

        # NB tests per competitor (RMSE and MAE on outer folds)
        comps = [e for e in exp_ids if e != ctrl]
        rows = []
        pvals_rmse, labels_rmse = [], []
        pvals_mae, labels_mae = [], []

        ctrl_rmse = np.array(
            data[ctrl]["rmse_folds"] if data[ctrl]["rmse_folds"] is not None else [],
            dtype=float,
        )
        ctrl_mae = np.array(
            data[ctrl]["mae_folds"] if data[ctrl]["mae_folds"] is not None else [],
            dtype=float,
        )

        if ctrl_rmse.size != args.k or ctrl_mae.size != args.k:
            f.write(
                "WARNING: Control model missing complete fold arrays; NB testing cannot proceed.\n"
            )
        else:
            for comp in comps:
                comp_rmse = data[comp]["rmse_folds"]
                comp_mae = data[comp]["mae_folds"]
                if comp_rmse is None or comp_mae is None:
                    continue
                comp_rmse = np.array(comp_rmse, dtype=float)
                comp_mae = np.array(comp_mae, dtype=float)
                if comp_rmse.size != args.k or comp_mae.size != args.k:
                    continue

                # diffs = competitor - control (losses); control better ⇒ positive mean
                d_rmse = comp_rmse - ctrl_rmse
                d_mae = comp_mae - ctrl_mae

                t_rmse, se_rmse, mean_rmse = nb_corrected_t(d_rmse, k=args.k)
                t_mae, se_mae, mean_mae = nb_corrected_t(d_mae, k=args.k)

                df = args.k - 1
                # RIGHT (upper tail): control better ⇢ mean(comp - ctrl) > 0
                p_rmse = float(tdist.sf(t_rmse, df=df))  # sf = 1 - cdf
                p_mae = float(tdist.sf(t_mae, df=df))

                pvals_rmse.append(p_rmse)
                labels_rmse.append(f"{ctrl} vs {comp}")
                pvals_mae.append(p_mae)
                labels_mae.append(f"{ctrl} vs {comp}")

                row = {
                    "comparison": f"{ctrl} vs {comp}",
                    "mean_diff_RMSE(comp-ctrl)": mean_rmse,
                    "t_NB_RMSE": t_rmse,
                    "p_one_sided_RMSE": p_rmse,
                    "mean_diff_MAE(comp-ctrl)": mean_mae,
                    "t_NB_MAE": t_mae,
                    "p_one_sided_MAE": p_mae,
                }

                if args.print_ci:
                    tcrit = float(tdist.ppf(1 - args.alpha / 2, df=df))
                    row["NB_CI_RMSE_low"] = mean_rmse - tcrit * se_rmse
                    row["NB_CI_RMSE_high"] = mean_rmse + tcrit * se_rmse
                    row["NB_CI_MAE_low"] = mean_mae - tcrit * se_mae
                    row["NB_CI_MAE_high"] = mean_mae + tcrit * se_mae

                rows.append(row)

        df_rows = pd.DataFrame(rows)
        if not df_rows.empty:
            f.write("--- NB-corrected t (outer folds) per competitor ---\n")
            f.write(
                df_rows.to_string(index=False, float_format=lambda x: f"{x:.6f}")
                + "\n\n"
            )
        else:
            f.write(
                "No comparable competitors with complete fold arrays. Nothing to test.\n\n"
            )

        # Holm across competitors per metric family
        def holm_table(pvals: List[float], labels: List[str], title: str):
            if not pvals:
                f.write(f"{title}\n(no p-values)\n\n")
                return
            reject, p_adj, _, _ = multipletests(
                pvals=pvals, alpha=args.alpha, method="holm"
            )
            df = pd.DataFrame(
                {
                    "comparison": labels,
                    "p_raw": pvals,
                    "p_holm": p_adj,
                    "Significant": reject.astype(bool),
                }
            )
            f.write(title + "\n")
            f.write(
                df.sort_values("p_raw").to_string(
                    index=False, float_format=lambda x: f"{x:.6f}"
                )
                + "\n\n"
            )

        holm_table(
            pvals_rmse, labels_rmse, "--- Holm-adjusted p-values (RMSE family) ---"
        )
        holm_table(
            pvals_mae, labels_mae, "--- Holm-adjusted p-values (MAE family)  ---"
        )

        f.write("=" * 110 + "\nNotes:\n")
        f.write(
            "• Tests are within-dataset, one-sided for control superiority, on outer-fold differences with Nadeau–Bengio SE correction (df = k-1).\n"
        )
        f.write(
            "• Holm controls family-wise error across competitors per metric family.\n"
        )
        f.write(
            "• Held-out Test metrics above are for context only; no fold-based omnibus tests are used.\n"
        )

    print(f"Report successfully written to: {report_path}")


if __name__ == "__main__":
    main()
