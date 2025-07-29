import torch
import numpy as np
from rdkit import Chem
import networkx as nx
from collections import defaultdict
from torch_geometric.data import Data
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE


def compressed_topsignal_graph_from_smiles(
    smile: str, y_val: int, topk_lap: int = 5
) -> Data | None:
    try:
        # 1) Abstract complex
        pg = PolyatomicGeometrySMILE(smile=smile, mode="abstract")
        ac = pg.smiles_to_geom_complex()
        assert isinstance(ac, AbstractComplex)

        # 2) RDKit molecule
        mol = Chem.MolFromSmiles(smile)  # type: ignore
        if mol is None:
            return None

        # 3) Node features: chain0 value + RDKit descriptors
        chains = ac.get_raw_k_chains()
        chain0 = chains.get("chain_0", [])
        atom_types = [6, 7, 8, 15, 16, 17]
        hyb_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
        ]
        node_feats = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            # fallback if chain0 shorter than atom count
            c0 = float(chain0[idx]) if idx < len(chain0) else 0.0
            feats = [c0]
            feats += one_hot(atom.GetAtomicNum(), atom_types)
            feats += one_hot(atom.GetHybridization(), hyb_types)
            feats += [
                float(atom.GetDegree()),
                float(atom.GetIsAromatic()),
                float(atom.GetFormalCharge()),
            ]
            node_feats.append(feats)
        x = torch.tensor(node_feats, dtype=torch.float32)
        n = x.size(0)  # use number of atoms for all subsequent node counts

        # 4) Edges: abstract bonds + RDKit fallback
        sk = ac.get_skeleta().get("molecule_skeleta", [[]])[0]
        zero = next((lst for dim, lst in sk if dim == "0"), [])
        node_ids = [next(iter(fz))[0] for fz in zero]
        atom_map = defaultdict(list)
        for i, nid in enumerate(node_ids):
            symbol = nid.split("_")[0]
            atom_map[symbol].append(i)

        edge_index_list, edge_attr_list = [], []
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        for a1, a2, (btype, order) in ac.get_bonds():
            bt_val = getattr(Chem.rdchem.BondType, btype, None)
            for i in atom_map.get(a1, []):
                for j in atom_map.get(a2, []):
                    if i < n and j < n:
                        edge_index_list += [[i, j], [j, i]]
                        attr = one_hot(bt_val, bond_types) + [float(order), 0.0]
                        edge_attr_list += [attr, attr]
        if not edge_index_list:
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index_list += [[i, j], [j, i]]
                attr = one_hot(bond.GetBondType(), bond_types)
                attr += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
                edge_attr_list += [attr, attr]

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        # 5) Topology features: centrality + SPD
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edge_index_list)
        cent = nx.closeness_centrality(G)
        spd = dict(nx.all_pairs_shortest_path_length(G))
        cent_vec = [cent.get(i, 0.0) for i in range(n)]
        spd_vec = [
            sum(d.values()) / max(len(d), 1) for d in (spd.get(i, {}) for i in range(n))
        ]
        cent_t = torch.tensor(cent_vec, dtype=torch.float32).view(n, 1)
        spd_t = torch.tensor(spd_vec, dtype=torch.float32).view(n, 1)
        x = torch.cat([x, cent_t, spd_t], dim=1)

        # print("MANAGED TO CONCAT?")

        # 6) Graph-level features: chain stats + laplacians
        g_stats, lap_feats = [], []
        for k, arr in chains.items():
            if k == "chain_0":
                continue
            a = np.array(arr, dtype=np.float32)
            g_stats += [a.mean(), a.std()]

        # print("COMPUTED GRAP STATS")

        for grp in ac.get_laplacians().get("molecule_laplacians", []):
            recs = grp if isinstance(grp, list) else [grp]
            for _, mat in recs:
                # use dense eigen solver to avoid ARPACK issues
                M = np.array(mat, dtype=np.float32)
                # compute eigenvalues of symmetric Laplacian
                try:
                    eigs = np.linalg.eigvalsh(M)
                except Exception:
                    eigs = np.zeros(M.shape[0], dtype=np.float32)
                # take smallest non-zero eigenvalues (skip the first zero)
                nonzero = eigs[eigs > 1e-6]
                vals = nonzero[:topk_lap] if len(nonzero) >= topk_lap else nonzero
                # pad to exactly topk_lap
                if len(vals) < topk_lap:
                    vals = np.pad(vals, (0, topk_lap - len(vals)))
                lap_feats += list(vals)

        # --- 7) Spectral k-chains stats ---
        spectral = ac.get_spectral_k_chains()
        spec_feats = []
        for arr in spectral.values():
            a = np.array(arr, dtype=np.float32)
            spec_feats += [a.mean(), a.std()]

        # --- 8) Betti numbers (components & cycles) ---
        b0 = nx.number_connected_components(G)
        b1 = sum(
            len(nx.cycle_basis(G.subgraph(comp))) for comp in nx.connected_components(G)
        )

        # --- 9) Assemble graph_feats ---
        all_feats = g_stats + lap_feats + spec_feats + [float(b0), float(b1)]
        graph_feats = torch.tensor(all_feats, dtype=torch.float32)

        # print("managed to feat?")

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_feats = graph_feats
        data.y = torch.tensor([y_val], dtype=torch.float)
        # print(f"SUCCESS for : {smile}")
        return data
    except Exception as e:
        # print(f"Failed {smile}: {e}")
        return None


def one_hot(val, choices):
    return [1.0 if val == c else 0.0 for c in choices]
