"""
Convert the plain-text summary (like the one you pasted) into a LaTeX table.

Features
- Parses blocks like:  --- Dataset: QM9 ---
- Reads Val metrics as "mean ± std"
- Reads Test metrics as either "mean [low, high]" (CI) or "mean ± std"
- Option --test-intervals {ci, pm}:
    ci  -> keep "mean [low, high]" strings (uses text columns for the 2 Test cols)
    pm  -> convert CI to ± half-width to match siunitx S columns
- Multirow per dataset; booktabs rules; siunitx S columns; optional renaming & bolding

Usage
    python latex_table_from_txt.py \
        --input results.txt --output table.tex \
        --test-intervals pm \
        --rename "polyatomic=PACTNet (ECC)" \
        --bold-contains "PACTNet" \
        --val-dec 3 --test-dec 4
"""

import argparse
import re
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert TXT performance summary to LaTeX table (booktabs + siunitx + multirow)."
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input TXT file")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output .tex file")
    p.add_argument(
        "--caption",
        default="Comprehensive performance comparison across all datasets and models.",
        help="LaTeX caption",
    )
    p.add_argument("--label", default="tab:full_results", help="LaTeX label")
    p.add_argument(
        "--val-dec",
        type=int,
        default=3,
        help="Decimal places for Val metrics (mean & std)",
    )
    p.add_argument(
        "--test-dec",
        type=int,
        default=4,
        help="Decimal places for Test metrics (mean & std)",
    )
    p.add_argument(
        "--no-fixed-decimals",
        action="store_true",
        help="Use raw decimals as provided (don't round to fixed places)",
    )
    p.add_argument(
        "--table-formats",
        nargs=4,
        default=["2.3(4)", "2.3(4)", "2.4(4)", "2.4(4)"],
        help="siunitx table-format for Val RMSE, Val MAE, Test RMSE, Test MAE",
    )
    p.add_argument(
        "--font-size", default="\\small", help="LaTeX font size inside the table"
    )
    p.add_argument("--width", default="\\textwidth", help="Width for \\resizebox")
    p.add_argument(
        "--no-resize", action="store_true", help="Disable \\resizebox wrapper"
    )
    p.add_argument(
        "--booktabs", action="store_true", default=True, help="Use booktabs rules"
    )
    p.add_argument(
        "--no-booktabs",
        dest="booktabs",
        action="store_false",
        help="Disable booktabs rules",
    )
    p.add_argument(
        "--test-intervals",
        choices=["ci", "pm"],
        default="pm",
        help="For Test metrics with CIs: keep CIs (ci) or convert to ± half-width (pm)",
    )
    p.add_argument(
        "--bold-contains",
        default=None,
        help="Regex to bold any row where model/rep cell matches",
    )
    p.add_argument(
        "--rename",
        nargs="*",
        default=[],
        help='Rename patterns like old=new (regex on the "Model (Rep.)" cell)',
    )
    p.add_argument("--dataset-order", nargs="*", help="Optional explicit dataset order")
    p.add_argument(
        "--sort-by",
        nargs="*",
        default=None,
        help="Sort keys within each dataset, e.g., --sort-by model representation",
    )
    p.add_argument(
        "--ascending",
        nargs="*",
        type=int,
        help="Ascending flags matching --sort-by, e.g. 1 0",
    )
    return p.parse_args()


def fmt_unc(mean, std, fixed_decimals: bool, dec_places: int) -> str:
    if pd.isna(mean) or pd.isna(std):
        return r"\textemdash"
    if fixed_decimals:
        return f"{float(mean):.{dec_places}f} \\pm {float(std):.{dec_places}f}"

    # keep raw, but tidy trailing zeros
    def tidy(x):
        s = f"{x}"
        if "e" in s or "E" in s:
            return s
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    return f"{tidy(mean)} \\pm {tidy(std)}"


def build_model_rep(name: str) -> tuple[str, str, str]:
    """
    Split 'gat_ecfp' -> ('GAT', 'ECFP', 'GAT (ECFP)')
    If no underscore, rep is ''.
    """
    name = name.strip()
    model, rep = name, ""
    if "_" in name:
        model, rep = name.split("_", 1)
    # Pretty-up casing: GAT -> GAT, selfies->SELFIES, etc.
    model_fmt = model.upper() if model.isalpha() else model
    rep_fmt = rep.upper() if rep else ""
    label = f"{model_fmt} ({rep_fmt})" if rep_fmt else model_fmt
    return model_fmt, rep_fmt, label


def apply_renames(s: str, mapping: dict) -> str:
    for k, v in mapping.items():
        s = re.sub(k, v, s)
    return s


def parse_metric(cell: str):
    """
    Returns dict with possible keys: mean, std, ci_low, ci_high
    Accepts:
      - '1.234 ± 0.056'
      - '1.234 [1.111, 1.345]'
      - '1.234'
    """
    s = cell.strip()
    m = re.match(r"([+-]?\d+(?:\.\d+)?)\s*±\s*([+-]?\d+(?:\.\d+)?)", s)
    if m:
        return {
            "mean": float(m.group(1)),
            "std": float(m.group(2)),
            "ci_low": None,
            "ci_high": None,
        }
    m = re.match(
        r"([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
        s,
    )
    if m:
        return {
            "mean": float(m.group(1)),
            "std": None,
            "ci_low": float(m.group(2)),
            "ci_high": float(m.group(3)),
        }
    m = re.match(r"([+-]?\d+(?:\.\d+)?)$", s)
    if m:
        return {"mean": float(m.group(1)), "std": None, "ci_low": None, "ci_high": None}
    return {"mean": None, "std": None, "ci_low": None, "ci_high": None}


def parse_txt(path: Path) -> pd.DataFrame:
    """
    Parse the text file structure you showed into a tidy DataFrame.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Find dataset blocks
    blocks = []
    for m in re.finditer(r"---\s*Dataset:\s*(.+?)\s*---", text):
        blocks.append((m.start(), m.group(1).strip()))
    rows = []
    for i, (pos, dataset) in enumerate(blocks):
        start = pos
        end = blocks[i + 1][0] if i + 1 < len(blocks) else len(text)
        body = text[start:end]

        # Capture the table lines after the header row with pipes
        table_lines = []
        after_header = False
        for line in body.splitlines():
            if re.search(r"\|\s*Val RMSE", line):
                after_header = True
                continue
            if after_header:
                if line.strip().startswith("--- Statistical"):
                    break
                if re.match(r"\s*$", line):
                    break
                # skip dashed separators
                if re.match(r"[-\s]{5,}$", line.replace("|", "")):
                    continue
                if "|" in line:
                    table_lines.append(line)

        for line in table_lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            name = parts[0]
            val_rmse = parse_metric(parts[1])
            val_mae = parse_metric(parts[2])
            test_rmse = parse_metric(parts[3])
            test_mae = parse_metric(parts[4])
            model, rep, label = build_model_rep(name)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "representation": rep,
                    "label": label,  # "Model (Rep.)" cell (pre-rename)
                    "val_rmse_mean": val_rmse["mean"],
                    "val_rmse_std": val_rmse["std"],
                    "val_mae_mean": val_mae["mean"],
                    "val_mae_std": val_mae["std"],
                    "test_rmse_mean": test_rmse["mean"],
                    "test_rmse_std": test_rmse["std"],
                    "test_rmse_ci_low": test_rmse["ci_low"],
                    "test_rmse_ci_high": test_rmse["ci_high"],
                    "test_mae_mean": test_mae["mean"],
                    "test_mae_std": test_mae["std"],
                    "test_mae_ci_low": test_mae["ci_low"],
                    "test_mae_ci_high": test_mae["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    df = parse_txt(args.input)

    # Optional sorting & dataset order
    if args.sort_by:
        asc = (
            [True] * len(args.sort_by)
            if args.ascending is None
            else [bool(int(a)) for a in args.ascending]
        )
        df = df.sort_values(by=args.sort_by, ascending=asc)
    if args.dataset_order:
        cat = pd.Categorical(df["dataset"], categories=args.dataset_order, ordered=True)
        df = df.assign(_dataset=cat).sort_values("_dataset").drop(columns="_dataset")

    # Renaming map
    rename_map = dict(kv.split("=", 1) for kv in args.rename) if args.rename else {}
    bold_re = (
        re.compile(args.bold_contains) if args.bold_contains else None
    )  # noqa: E999 (hyphen in attribute name)
    # The above line will fail because of hyphen; fix properly:
    bold_re = re.compile(args.bold_contains) if args.bold_contains else None

    # Column spec (if CI mode, last two are text columns)
    if args.test_intervals == "ci":
        colspec = (
            "@{}ll "
            + " ".join(
                [
                    f"S[table-format={args.table_formats[0]}]",
                    f"S[table-format={args.table_formats[1]}]",
                    "l",
                    "l",
                ]
            )
            + "@{}"
        )
    else:
        colspec = (
            "@{}ll "
            + " ".join([f"S[table-format={tf}]" for tf in args.table_formats])
            + "@{}"
        )

    # Start LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    if args.font_size:
        lines.append(f"{args.font_size} % Font size")
    lines.append(r"\caption{" + args.caption + r"}")
    lines.append(r"\label{" + args.label + r"}")
    lines.append(r"% siunitx settings")
    lines.append(r"\sisetup{separate-uncertainty, table-align-text-post=false}")

    inner_begin = r"\begin{tabular}{" + colspec + r"}"
    inner_end = r"\end{tabular}"

    if args.no_resize:
        lines.append(inner_begin)
    else:
        lines.append(r"\resizebox{" + args.width + r"}{!}{" + inner_begin)

    if args.booktabs:
        lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Model (Rep.)} & {Val RMSE} & {Val MAE} & {Test RMSE} & {Test MAE} \\"
    )
    if args.booktabs:
        lines.append(r"\midrule")

    # Emit grouped rows with \multirow
    for dataset, g in df.groupby("dataset", sort=False):
        n = len(g)
        first = True
        for _, row in g.iterrows():
            # Model/Rep cell with renames + optional bold
            cell_model = apply_renames(row["label"], rename_map)
            do_bold = bool(bold_re and bold_re.search(cell_model)) if bold_re else False

            # Val metrics (always ± form in your text)
            val_rmse = fmt_unc(
                row["val_rmse_mean"],
                row["val_rmse_std"],
                fixed_decimals=not args.no_fixed_decimals,
                dec_places=args.val_dec,
            )
            val_mae = fmt_unc(
                row["val_mae_mean"],
                row["val_mae_std"],
                fixed_decimals=not args.no_fixed_decimals,
                dec_places=args.val_dec,
            )

            # Test metrics
            def ci_or_pm(mean, std, lo, hi):
                if args.test_intervals == "ci" and (lo is not None and hi is not None):
                    return f"{mean} [{lo}, {hi}]"
                if std is None and (lo is not None and hi is not None):
                    std = (hi - lo) / 2.0
                return fmt_unc(
                    mean,
                    std,
                    fixed_decimals=not args.no_fixed_decimals,
                    dec_places=args.test_dec,
                )

            test_rmse = ci_or_pm(
                row["test_rmse_mean"],
                row["test_rmse_std"],
                row["test_rmse_ci_low"],
                row["test_rmse_ci_high"],
            )
            test_mae = ci_or_pm(
                row["test_mae_mean"],
                row["test_mae_std"],
                row["test_mae_ci_low"],
                row["test_mae_ci_high"],
            )

            parts = []
            if first:
                parts.append(rf"\multirow{{{n}}}{{*}}{{{dataset}}}")
                first = False
            else:
                parts.append("")  # empty dataset cell

            if do_bold:
                parts.append(rf"\bfseries {cell_model}")
                parts.append(rf"\bfseries {val_rmse}")
                parts.append(rf"\bfseries {val_mae}")
                parts.append(rf"\bfseries {test_rmse}")
                parts.append(rf"\bfseries {test_mae}")
            else:
                parts.append(cell_model)
                parts.append(val_rmse)
                parts.append(val_mae)
                parts.append(test_rmse)
                parts.append(test_mae)

            lines.append(" & ".join(parts) + r" \\")

    if args.booktabs:
        lines.append(r"\bottomrule")
    lines.append(inner_end)
    if not args.no_resize:
        lines.append("}")
    lines.append(r"\end{table}")

    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
