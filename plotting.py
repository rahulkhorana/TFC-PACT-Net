import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_parity(y_true, y_pred, title, save_path: Path):
    """
    Generates and saves a parity plot (predicted vs. true values).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate metrics for plot annotation
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor="k", s=50, ax=ax)

    # y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Ideal (y=x)")
    ax.set_aspect("equal")
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])

    # Labels and title
    ax.set_xlabel("True Values", fontsize=14)
    ax.set_ylabel("Predicted Values", fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)

    # Add metrics to the plot
    ax.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"INFO: Plot saved to {save_path}")
