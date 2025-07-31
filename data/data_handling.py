import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

# --- Assume these are imported from your project ---
from data.loaders import load_dataset
from data.featurize import (
    smiles_to_graph,
    selfies_to_graph,
    ecfp_to_graph,
    smiles_for_gp,
    selfies_for_gp,
    ecfp_for_gp,
)
from data.polyatomic_featurize import compressed_topsignal_graph_from_smiles
from models.gnn import GCN, GIN, GAT, GraphSAGE
from models.polyatomic import PolyatomicNet

# --- Configuration Dictionaries ---
# Representation functions for GNN
REPRESENTATIONS = {
    "smiles": smiles_to_graph,
    "selfies": selfies_to_graph,
    "ecfp": ecfp_to_graph,
    "polyatomic": compressed_topsignal_graph_from_smiles,
}

# Feature functions for GP (numeric vectors)
GP_FEATURIZERS = {
    "smiles": smiles_for_gp,
    "selfies": selfies_for_gp,
    "ecfp": ecfp_for_gp,
}

GNN_MODELS = {
    "gcn": GCN,
    "gin": GIN,
    "gat": GAT,
    "sage": GraphSAGE,
    "polyatomic": PolyatomicNet,  # Custom GNN for polyatomic complexes
}

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass


def featurize_dataset_parallel(X, y, featurizer, n_jobs=None):
    """Your provided parallel featurization function."""
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)

    if featurizer.__name__ == "compressed_topsignal_graph_from_smiles":
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(featurizer)(xi, yi) for xi, yi in zip(X, y)
        )
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(featurizer)(xi) for xi in X
        )

    # Filter out None results and attach the original y-value
    data_list = []
    # This robustly handles cases where some featurizations might fail
    for i, res in enumerate(results):
        if res is not None:
            res.y = torch.tensor([y[i]], dtype=torch.float)
            data_list.append(res)

    return data_list


def prepare_and_load_data(args):
    """
    Performs the expensive featurization ONCE and caches the result.
    Subsequent runs will load the cached file instantly.
    """
    cache_dir = Path("precomputed_features")
    cache_dir.mkdir(exist_ok=True)
    train_cache_file = cache_dir / f"{args.dataset}_{args.rep}_train.pt"
    test_cache_file = cache_dir / f"{args.dataset}_{args.rep}_test.pt"

    if train_cache_file.exists() and test_cache_file.exists():
        print(
            f"INFO: Loading pre-featurized data from cache for dataset '{args.dataset}'..."
        )
        train_graphs = torch.load(train_cache_file)
        test_graphs = torch.load(test_cache_file)
        return train_graphs, test_graphs

    print("INFO: No cached data found. Starting one-time featurization process...")
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    featurizer = REPRESENTATIONS[args.rep]

    print("Featurizing training set (this may take a while)...")
    train_graphs = featurize_dataset_parallel(X_train, y_train, featurizer)
    torch.save(train_graphs, train_cache_file)
    print(f"Saved featurized training data to {train_cache_file}")

    print("Featurizing test set...")
    test_graphs = featurize_dataset_parallel(X_test, y_test, featurizer)
    torch.save(test_graphs, test_cache_file)
    print(f"Saved featurized test data to {test_cache_file}")

    return train_graphs, test_graphs


def get_model_instance(args, params, train_graphs):
    """Instantiates a model, handling the special case for polyatomic."""
    model_class = GNN_MODELS[args.model]
    sample_graph = train_graphs[0]

    if args.model == "polyatomic":
        from torch_geometric.utils import degree

        print("INFO: Calculating degree vector for polyatomic model...")
        loader = DataLoader(train_graphs, batch_size=params.get("batch_size", 128))
        deg = torch.zeros(32, dtype=torch.long)
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            bc = torch.bincount(d, minlength=deg.size(0))
            if bc.size(0) > deg.size(0):
                new_deg = torch.zeros(bc.size(0), dtype=torch.long)
                new_deg[: deg.size(0)] = deg
                deg = new_deg
            deg += bc
        return model_class(
            node_feat_dim=sample_graph.x.shape[1],
            edge_feat_dim=sample_graph.edge_attr.shape[1],
            graph_feat_dim=sample_graph.graph_feats.shape[0],
            hidden_dim=params["hidden_dim"],
            deg=deg,
        )
    else:
        return model_class(
            in_channels=sample_graph.num_node_features,
            hidden_channels=params["hidden_dim"],
            out_channels=1,
        )
