import pandas as pd
from sklearn.model_selection import train_test_split
import importlib.resources as pkg_resources
import polyatomic_complexes
import numpy as np
from typing import Tuple


def load_dataset(name) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    # fixed random seed for reproducibility, and datasets split
    Load dataset and split into training and test sets.
    Args:
        name (str): Name of the dataset to load. Options are 'esol', 'freesolv', 'lipophil'.
    Returns:
        X_train (np.ndarray): Training set features.
        X_test (np.ndarray): Test set features.
        y_train (np.ndarray): Training set targets.
        y_test (np.ndarray): Test set targets.
    """
    if name.lower() == "esol":
        data_path = (
            pkg_resources.files("polyatomic_complexes.dataset.esol") / "ESOL.csv"
        )
        df = pd.read_csv(str(data_path))
        smiles = df["smiles"]
        targets = df["measured log solubility in mols per litre"]
        target_col = "measured log solubility in mols per litre"
    elif name.lower() == "freesolv":
        data_path = (
            pkg_resources.files("polyatomic_complexes.dataset.free_solv")
            / "FreeSolv.csv"
        )
        df = pd.read_csv(str(data_path))
        smiles = df["smiles"]
        targets = df["expt"]  # expt
        target_col = "expt"
    elif name.lower() == "lipophil":
        data_path = (
            pkg_resources.files("polyatomic_complexes.dataset.lipophilicity")
            / "Lipophilicity.csv"
        )
        df = pd.read_csv(str(data_path))
        smiles = df["smiles"]
        targets = df["exp"]  # exp
        target_col = "exp"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df.dropna(subset=["smiles", target_col], inplace=True)
    smiles = df["smiles"]
    targets = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        smiles, targets, test_size=0.2, random_state=42
    )
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
