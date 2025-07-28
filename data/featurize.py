import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import selfies as sf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

mfpgen = GetMorganGenerator(
    radius=2,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=False,
    includeRingMembership=True,
    countBounds=None,
    fpSize=2048,
    atomInvariantsGenerator=None,
    bondInvariantsGenerator=None,
    includeRedundantEnvironments=False,
)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        return None
    return mol_to_graph(mol)


def selfies_to_graph(selfies_str):
    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if mol is None:
            return None
        return mol_to_graph(mol)
    except:
        return None


def ecfp_to_graph(fp, radius=2):
    node_indices = [i for i, bit in enumerate(fp) if bit == 1]
    x = torch.eye(len(node_indices))

    edge_index = []
    for i in range(len(node_indices)):
        for j in range(i + 1, len(node_indices)):
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.ones((edge_index.size(1), 1))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mol_to_graph(mol):
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(
            [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetIdx(),
            ]
        )
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
        btype = bond.GetBondTypeAsDouble()
        edge_attr.append([btype])
        edge_attr.append([btype])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_for_gp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        return np.zeros(mfpgen.GetNumBits(), dtype=np.float32)
    arr = mfpgen.GetFingerprintAsNumPy(mol)
    return arr.astype(np.float32)


def selfies_for_gp(selfies_str, radius=2, n_bits=2048):
    try:
        smiles = sf.decoder(selfies_str)
        assert isinstance(smiles, str)
        return smiles_for_gp(smiles)
    except:
        return np.zeros(n_bits)


def ecfp_for_gp(smiles_str: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles_str)  # type: ignore
    if mol is None:
        return np.zeros(mfpgen.GetNumBits(), dtype=np.float32)
    return mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32)


def graph_native_loader(graph_list, batch_size=32, shuffle=True):
    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)
