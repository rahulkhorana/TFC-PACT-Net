import pandas as pd
from sklearn.model_selection import train_test_split
import importlib.resources as pkg_resources
import polyatomic_complexes
import numpy as np
from typing import Tuple
from pathlib import Path


def load_dataset(name) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name.lower() == "esol":
        data_path = (
            pkg_resources.files("polyatomic_complexes.dataset.esol") / "ESOL.csv"
        )
        df = pd.read_csv(str(data_path))
        target_col = "measured log solubility in mols per litre"
    elif name.lower() == "freesolv":
        data_path = (
            pkg_resources.files("polyatomic_complexes.dataset.free_solv")
            / "FreeSolv.csv"
        )
        df = pd.read_csv(str(data_path))
        target_col = "expt"
    elif name.lower() == "lipophil":
        data_path = (
            pkg_resources.files("polyatomic_complexes.dataset.lipophilicity")
            / "Lipophilicity.csv"
        )
        df = pd.read_csv(str(data_path))
        target_col = "exp"
    elif name.lower() == "qm8":
        data_path = (
            Path(__file__).parent.parent / "benchmark_csv/qm8_subset.csv".__str__()
        )
        df = pd.read_csv(data_path)
        target_col = "f1-CAM"
    elif name.lower() == "qm9":
        data_path = (
            Path(__file__).parent.parent / "benchmark_csv/qm9_subset.csv".__str__()
        )
        df = pd.read_csv(data_path)
        target_col = "cv"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df.dropna(subset=["smiles", target_col], inplace=True)
    smiles = df["smiles"]
    targets = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        smiles, targets, test_size=0.2, random_state=42
    )
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
