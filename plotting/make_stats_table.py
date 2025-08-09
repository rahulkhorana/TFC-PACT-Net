#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse *_NB_single_dataset.txt reports and emit LaTeX tables.

Now supports:
  • Full "everything" longtable (as before)
  • Option B: TWO COMPACT portrait tables (RMSE table + MAE table)

Examples
--------
# Make compact + full tables for all reports in a dir
python3 make_nb_tables_from_reports.py --in ./plotting --out ./plotting/tex

# Only compact tables
python3 make_nb_tables_from_reports.py --in ./plotting --out ./plotting/tex --compact_only

# Specific files
python3 make_nb_tables_from_reports.py \
  --files ./plotting/qm9_NB_single_dataset.txt ./plotting/boilingpoint_NB_single_dataset.txt \
  --out ./plotting/tex --compact_only
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def esc_tex(s: str) -> str:
    return s.replace("_", r"\_")


def parse_header(lines: List[str]) -> Dict[str, str]:
    out = {}
    for ln in lines[:60]:
        m = re.search(r"^Dataset:\s*(.+?)\s+—", ln)
        if m:
            out["dataset"] = m.group(1).strip()
        m = re.search(r"^Control exp_id:\s*(.+)$", ln)
        if m:
            out["control"] = m.group(1).strip()
        m = re.search(r"^k folds:\s*(\d+),\s*alpha:\s*(" + NUM + r")", ln)
        if m:
            out["k"] = int(m.group(1))
            out["alpha"] = float(m.group(2))
    return out


def find_block(lines: List[str], start_pat: str) -> Tuple[int, int]:
    start = -1
    pat = re.compile(start_pat)
    for i, ln in enumerate(lines):
        if pat.search(ln):
            start = i
            break
    if start < 0:
        return (-1, -1)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not lines[j].strip():
            end = j
            break
    return (start, end)


def parse_nb_block(lines: List[str]) -> pd.DataFrame:
    hdr_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"^\s*comparison\s+", ln) and "mean_diff_RMSE" in ln:
            hdr_idx = i
            break
    if hdr_idx is None:
        return pd.DataFrame()

    data_lines = []
    for ln in lines[hdr_idx + 1 :]:
        s = ln.rstrip()
        if not s:
            break
        if set(s) <= set("- "):
            continue
        data_lines.append(s)

    rows = []
    for ln in data_lines:
        # Flexible parse: "comparison  num num ... (10 nums total)"
        m = re.match(r"^\s*(.+?)\s+(" + NUM + r"(?:\s+" + NUM + r"){9})\s*$", ln)
        if not m:
            parts = re.split(r"\s{2,}", ln.strip())
            if len(parts) < 11:
                continue
            comp, *nums = parts
        else:
            comp = m.group(1).strip()
            nums = re.split(r"\s+", m.group(2).strip())
        if len(nums) != 10:
            continue
        vals = list(map(float, nums))
        rows.append(
            {
                "comparison": comp,
                "mean_diff_RMSE": vals[0],
                "t_NB_RMSE": vals[1],
                "p_raw_RMSE": vals[2],
                "mean_diff_MAE": vals[3],
                "t_NB_MAE": vals[4],
                "p_raw_MAE": vals[5],
                "CI_RMSE_low": vals[6],
                "CI_RMSE_high": vals[7],
                "CI_MAE_low": vals[8],
                "CI_MAE_high": vals[9],
            }
        )
    return pd.DataFrame(rows)


def parse_holm_block(lines: List[str], metric_label: str) -> pd.DataFrame:
    hdr = None
    for i, ln in enumerate(lines):
        if re.search(r"^\s*comparison\s+", ln) and "p_raw" in ln and "p_holm" in ln:
            hdr = i
            break
    if hdr is None:
        return pd.DataFrame()

    data_lines = []
    for ln in lines[hdr + 1 :]:
        s = ln.rstrip()
        if not s:
            break
        if set(s) <= set("- "):
            continue
        data_lines.append(s)

    rows = []
    for ln in data_lines:
        m = re.match(r"^\s*(.+?)\s+(" + NUM + r")\s+(" + NUM + r")\s+(\w+)\s*$", ln)
        if m:
            comp = m.group(1).strip()
            p_raw = float(m.group(2))  # not used here, but parsed ok
            p_holm = float(m.group(3))
            sig = m.group(4).strip().lower().startswith("t")
        else:
            parts = re.split(r"\s{2,}", ln.strip())
            if len(parts) < 4:
                continue
            comp = parts[0].strip()
            p_holm = float(parts[2])
            sig = parts[3].strip().lower().startswith("t")
        rows.append(
            {
                "comparison": comp,
                f"p_holm_{metric_label}": p_holm,
                f"sig_{metric_label}": sig,
            }
        )
    return pd.DataFrame(rows)


# ---------- FULL longtable (unchanged behavior) ----------


def write_full_longtable(meta: Dict[str, str], df: pd.DataFrame, outpath: Path):
    dataset = meta.get("dataset", "dataset")
    control = meta.get("control", "control")
    k = int(meta.get("k", 5))
    alpha = float(meta.get("alpha", 0.05))

    header = rf"""
% Auto-generated from NB_single_dataset report
% Requires: \usepackage{{booktabs,longtable,siunitx}}
\begin{{longtable}}{{@{{}}l
S S S S S S S
S
S S S S S S S
@{{}}}}
\caption{{Full NB-corrected one-sided tests (outer folds, $K={k}$, $\alpha={alpha}$) comparing control \texttt{{{esc_tex(control)}}} against each competitor on dataset \texttt{{{esc_tex(dataset)}}}.
Positive $\Delta$ (competitor $-$ control) favors the control (lower loss).
One-sided alternative $\mathbb{{E}}[\bar d^{{(L)}}] > 0$.
Holm step-down controls FWER per metric family (RMSE and MAE reported separately).
CIs are NB-style two-sided.}}
\label{{tab:{esc_tex(dataset)}_nb_full}}\\
\toprule
& \multicolumn{{7}}{{c}}{{\textbf{{RMSE family}}}} & &
\multicolumn{{7}}{{c}}{{\textbf{{MAE family}}}}\\
\cmidrule(lr){{2-8}}\cmidrule(lr){{10-16}}
Comparison
& {{\(\Delta\)RMSE}}
& {{$t_{{\text{{NB}}}}$}}
& {{p (raw)}}
& {{p\(_{{\text{{Holm}}}}\)}}
& {{\(\text{{Sig}}\)}}
& {{CI\(_{{\text{{low}}}}\)}}
& {{CI\(_{{\text{{high}}}}\)}}
& {{}} % spacer
& {{\(\Delta\)MAE}}
& {{$t_{{\text{{NB}}}}$}}
& {{p (raw)}}
& {{p\(_{{\text{{Holm}}}}\)}}
& {{\(\text{{Sig}}\)}}
& {{CI\(_{{\text{{low}}}}\)}}
& {{CI\(_{{\text{{high}}}}\)}}\\
\midrule
\endfirsthead
\toprule
& \multicolumn{{7}}{{c}}{{\textbf{{RMSE family}}}} & &
\multicolumn{{7}}{{c}}{{\textbf{{MAE family}}}}\\
\cmidrule(lr){{2-8}}\cmidrule(lr){{10-16}}
Comparison
& {{\(\Delta\)RMSE}}
& {{$t_{{\text{{NB}}}}$}}
& {{p (raw)}}
& {{p\(_{{\text{{Holm}}}}\)}}
& {{\(\text{{Sig}}\)}}
& {{CI\(_{{\text{{low}}}}\)}}
& {{CI\(_{{\text{{high}}}}\)}}
& {{}} % spacer
& {{\(\Delta\)MAE}}
& {{$t_{{\text{{NB}}}}$}}
& {{p (raw)}}
& {{p\(_{{\text{{Holm}}}}\)}}
& {{\(\text{{Sig}}\)}}
& {{CI\(_{{\text{{low}}}}\)}}
& {{CI\(_{{\text{{high}}}}\)}}\\
\midrule
\endhead
\midrule
\multicolumn{{16}}{{r}}{{\emph{{Continued on next page}}}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
"""
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"{esc_tex(r['comparison'])} & "
            f"{r['mean_diff_RMSE']:.6f} & {r['t_NB_RMSE']:.6f} & {r['p_raw_RMSE']:.6f} & {r['p_holm_RMSE']:.6f} & {1 if r['sig_RMSE'] else 0} & {r['CI_RMSE_low']:.6f} & {r['CI_RMSE_high']:.6f} & & "
            f"{r['mean_diff_MAE']:.6f} & {r['t_NB_MAE']:.6f} & {r['p_raw_MAE']:.6f} & {r['p_holm_MAE']:.6f} & {1 if r['sig_MAE'] else 0} & {r['CI_MAE_low']:.6f} & {r['CI_MAE_high']:.6f} \\\\"
        )
    outpath.write_text(
        header + "\n".join(lines) + "\n\\end{longtable}\n", encoding="utf-8"
    )


# ---------- COMPACT portrait tables (Option B) ----------


def write_compact_tables(meta: Dict[str, str], df: pd.DataFrame, outdir: Path):
    """
    Produce two small portrait tables:
      • <dataset>_nb_compact_rmse.tex
      • <dataset>_nb_compact_mae.tex

    Columns (per metric): Comparison, Δ, t_NB, CI_low, CI_high, p_Holm
    """
    dataset = meta.get("dataset", "dataset")
    control = meta.get("control", "control")
    k = int(meta.get("k", 5))

    common_preamble = r"""
% Auto-generated compact NB tables
% Requires in preamble: \usepackage{booktabs,tabularx,siunitx}
% \sisetup{round-mode=places,round-precision=3,scientific-notation=true}
"""

    # RMSE table
    rmse_header = rf"""{common_preamble}
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{RMSE: NB-corrected one-sided tests (outer folds, $K={k}$) on dataset \texttt{{{esc_tex(dataset)}}}; control \texttt{{{esc_tex(control)}}}.
Positive $\Delta$ (competitor $-$ control) favors control. Holm controls FWER.}}
\label{{tab:{esc_tex(dataset)}_nb_compact_rmse}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}l S S S S S@{{}}}}
\toprule
Comparison & {{\(\Delta\)RMSE}} & {{$t_{{\text{{NB}}}}$}} & {{CI\(_{{\text{{low}}}}\)}} & {{CI\(_{{\text{{high}}}}\)}} & {{p\(_{{\text{{Holm}}}}\)}}\\
\midrule
"""
    rmse_rows = []
    for _, r in df.sort_values("p_holm_RMSE").iterrows():
        rmse_rows.append(
            f"{esc_tex(r['comparison'])} & "
            f"{r['mean_diff_RMSE']:.3f} & {r['t_NB_RMSE']:.3f} & "
            f"{r['CI_RMSE_low']:.3f} & {r['CI_RMSE_high']:.3f} & {r['p_holm_RMSE']:.3g} \\\\"
        )
    rmse_tex = (
        rmse_header
        + "\n".join(rmse_rows)
        + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )
    (outdir / f"{dataset}_nb_compact_rmse.tex").write_text(rmse_tex, encoding="utf-8")

    # MAE table
    mae_header = rf"""{common_preamble}
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{MAE: NB-corrected one-sided tests (outer folds, $K={k}$) on dataset \texttt{{{esc_tex(dataset)}}}; control \texttt{{{esc_tex(control)}}}.
Positive $\Delta$ (competitor $-$ control) favors control. Holm controls FWER.}}
\label{{tab:{esc_tex(dataset)}_nb_compact_mae}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}l S S S S S@{{}}}}
\toprule
Comparison & {{\(\Delta\)MAE}} & {{$t_{{\text{{NB}}}}$}} & {{CI\(_{{\text{{low}}}}\)}} & {{CI\(_{{\text{{high}}}}\)}} & {{p\(_{{\text{{Holm}}}}\)}}\\
\midrule
"""
    mae_rows = []
    for _, r in df.sort_values("p_holm_MAE").iterrows():
        mae_rows.append(
            f"{esc_tex(r['comparison'])} & "
            f"{r['mean_diff_MAE']:.3f} & {r['t_NB_MAE']:.3f} & "
            f"{r['CI_MAE_low']:.3f} & {r['CI_MAE_high']:.3f} & {r['p_holm_MAE']:.3g} \\\\"
        )
    mae_tex = (
        mae_header
        + "\n".join(mae_rows)
        + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    )
    (outdir / f"{dataset}_nb_compact_mae.tex").write_text(mae_tex, encoding="utf-8")


def process_file(path: Path, outdir: Path, make_full: bool, make_compact: bool):
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta = parse_header(txt)

    # NB block
    nb_start, nb_end = find_block(
        txt, r"^\s*--- NB-corrected t \(outer folds\) per competitor ---"
    )
    if nb_start < 0:
        raise RuntimeError(f"NB block not found in {path}")
    df_nb = parse_nb_block(txt[nb_start:nb_end])
    if df_nb.empty:
        raise RuntimeError(f"NB table parsing failed in {path}")

    # Holm blocks
    hr_start, hr_end = find_block(
        txt, r"^\s*--- Holm-adjusted p-values \(RMSE family\)\s*---"
    )
    hm_start, hm_end = find_block(
        txt, r"^\s*--- Holm-adjusted p-values \(MAE family\)\s*---"
    )
    if hr_start < 0 or hm_start < 0:
        raise RuntimeError(f"Holm blocks not found in {path}")
    df_hr = parse_holm_block(txt[hr_start:hr_end], "RMSE")
    df_hm = parse_holm_block(txt[hm_start:hm_end], "MAE")

    # Merge Holm into NB
    df = df_nb.merge(df_hr, on="comparison", how="left").merge(
        df_hm, on="comparison", how="left"
    )

    # Write outputs
    dataset = meta.get("dataset", path.stem.replace("_NB_single_dataset", ""))
    outdir.mkdir(parents=True, exist_ok=True)

    if make_full:
        full_path = outdir / f"{dataset}_nb_full_summary_from_report.tex"
        # order for full table: by raw p (RMSE then MAE)
        df_full = df.sort_values(["p_raw_RMSE", "p_raw_MAE"]).reset_index(drop=True)
        write_full_longtable(meta, df_full, full_path)

    if make_compact:
        # For compact tables we sort within each family by Holm p
        write_compact_tables(meta, df, outdir)

    print(
        f"Processed: {path.name}  ->  {dataset} (full={make_full}, compact={make_compact})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="indir",
        type=str,
        default=None,
        help="Directory containing *_NB_single_dataset.txt files.",
    )
    ap.add_argument(
        "--files", nargs="*", default=None, help="Explicit list of report files."
    )
    ap.add_argument(
        "--out",
        dest="outdir",
        type=str,
        required=True,
        help="Output directory for .tex tables.",
    )
    ap.add_argument(
        "--compact_only",
        action="store_true",
        help="Only write the compact RMSE/MAE tables (skip the full longtable).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.files:
        files = [Path(f) for f in args.files]
    elif args.indir:
        files = sorted(Path(args.indir).glob("*_NB_single_dataset.txt"))
    else:
        raise SystemExit("Provide either --in DIR or --files file1 file2 ...")

    if not files:
        raise SystemExit("No *_NB_single_dataset.txt files found.")

    for f in files:
        process_file(f, outdir, make_full=not args.compact_only, make_compact=True)


if __name__ == "__main__":
    main()
