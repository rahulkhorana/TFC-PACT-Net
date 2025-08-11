import os
import re
from collections import defaultdict

LOG_ROOT = "logs"
SUMMARY_FILE = "benchmark_summary.txt"


def parse_ci_section(content):
    """Extracts Test RMSE, MAE, and their 95% CI intervals from log content."""
    rmse_match = re.search(
        r"Test RMSE: ([\d.]+)\s+\(95 % CI: ([\d.]+)[–-]([\d.]+)\)", content
    )
    mae_match = re.search(
        r"Test MAE\s*: ([\d.]+)\s+\(95 % CI: ([\d.]+)[–-]([\d.]+)\)", content
    )

    if not (rmse_match and mae_match):
        return None

    rmse, rmse_lo, rmse_hi = map(float, rmse_match.groups())
    mae, mae_lo, mae_hi = map(float, mae_match.groups())

    rmse_pm = (rmse_hi - rmse_lo) / 2
    mae_pm = (mae_hi - mae_lo) / 2

    return {"rmse": rmse, "rmse_pm": rmse_pm, "mae": mae, "mae_pm": mae_pm}


def extract_encoding(fname):
    for enc in ["ecfp", "smiles", "selfies"]:
        if enc in fname.lower():
            return enc
    return "unknown"


# Map benchmark name → list of rows
benchmark_tables = defaultdict(list)

for benchmark_dir in os.listdir(LOG_ROOT):
    if not benchmark_dir.endswith("-bench"):
        continue

    benchmark_name = benchmark_dir.replace("-bench", "")
    benchmark_path = os.path.join(LOG_ROOT, benchmark_dir)

    for model in os.listdir(benchmark_path):
        model_path = os.path.join(benchmark_path, model)
        if not os.path.isdir(model_path):
            continue

        for log_file in os.listdir(model_path):
            if not log_file.endswith(".txt"):
                continue

            encoding = extract_encoding(log_file)
            with open(os.path.join(model_path, log_file), "r") as f:
                content = f.read()

            parsed = parse_ci_section(content)
            if parsed is None:
                continue

            benchmark_tables[benchmark_name].append(
                {"model": model, "encoding": encoding, **parsed}
            )

# Write to file
with open(SUMMARY_FILE, "w") as f:

    def log(line=""):
        f.write(line + "\n")
        print(line)

    for benchmark, rows in sorted(benchmark_tables.items()):
        log(f"\n{benchmark.upper()} Benchmark Summary:\n")
        log(f"{'Model':<13} {'Encoding':<8} {'RMSE ±':<20} {'MAE ±':<20}")
        log("-" * 65)

        for row in sorted(rows, key=lambda r: (r["model"], r["encoding"])):
            log(
                f"{row['model']:<13} {row['encoding']:<8} "
                f"{row['rmse']:.4f} ± {row['rmse_pm']:.4f}     "
                f"{row['mae']:.4f} ± {row['mae_pm']:.4f}"
            )
