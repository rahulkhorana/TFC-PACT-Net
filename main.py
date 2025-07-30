import os
import argparse
import wandb
import numpy as np
import torch.nn as nn
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
from models.polyatomic import PolyatomicNet
from models.gnn import GCN, GIN, GAT, GraphSAGE
from models.gp import TanimotoGP as GPModel
from training.train_eval import (
    train_gp_model,
    eval_gp_model,
    train_gnn_model,
    eval_gnn_model,
    train_polyatomic,
    evaluate_polyatomic,
    k_fold_eval,
)
import torch
from torch_geometric.loader import DataLoader
from torch.amp.grad_scaler import GradScaler

import multiprocessing

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass  # start method already set — safe to ignore

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


def featurize_dataset(X, y, featurizer):
    data_list = []
    for xi, yi in zip(X, y):
        g = featurizer(xi)
        if g is not None:  # ✅ skip invalid conversions
            g.y = torch.tensor([yi], dtype=torch.float)
            data_list.append(g)
    if len(data_list) == 0:
        raise ValueError("All graph inputs were invalid! Check featurizer or dataset.")
    return data_list


def featurize_dataset_parallel(X, y, featurizer, n_jobs=None):
    import multiprocessing
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(featurizer)(smiles, y) for smiles, y in zip(X, y)
    )
    return [g for g in results if g is not None]


def run_gp(args):
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    gp_feat = GP_FEATURIZERS[args.rep]

    # Featurize for GP
    X_train = np.stack([gp_feat(x) for x in X_train]).astype(np.float32)
    X_test = np.stack([gp_feat(x) for x in X_test]).astype(np.float32)

    # Define train/eval
    def train_fn(X_tr, y_tr, log_file):
        model = GPModel()
        return train_gp_model(model, X_tr, y_tr, log_file)

    def eval_fn(model, X_te, y_te, log_file, scaler, return_preds=False):
        return eval_gp_model(
            model, X_te, y_te, log_file, scaler, return_preds=return_preds
        )

    wandb.init(
        project="chem-reps", name=f"GP-{args.rep}-{args.dataset}", config=vars(args)
    )

    # Train using K-Fold CV, evaluate on held-out test set with CI
    fold_metrics, test_metrics = k_fold_eval(
        train_fn=train_fn,
        eval_fn=eval_fn,
        X_train=X_train,
        y_train=y_train,
        model_name="gp",
        rep_name=args.rep,
        dataset_name=args.dataset,
        X_test=X_test,
        y_test=y_test,
    )

    wandb.log({"final_test": test_metrics})
    wandb.finish()


def run_gnn(args):
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    featurizer = REPRESENTATIONS[args.rep]

    def train_fn(X_tr, y_tr, log_file):
        data_list = featurize_dataset(X_tr, y_tr, featurizer)
        loader = DataLoader(data_list, batch_size=32, shuffle=True)
        model_cls = GNN_MODELS[args.model]
        model = model_cls(data_list[0].num_node_features, 64, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_gnn_model(model, loader, optimizer, log_file)
        return model

    def eval_fn(model, X_te, y_te, log_file, scaler, return_preds=False):
        data_list = featurize_dataset(X_te, y_te, featurizer)
        loader = DataLoader(data_list, batch_size=32)
        return eval_gnn_model(
            model, loader, log_file, scaler, return_preds=return_preds
        )

    wandb.init(
        project="chem-reps",
        name=f"GNN-{args.model}-{args.rep}-{args.dataset}",
        config=vars(args),
    )

    fold_metrics, test_metrics = k_fold_eval(
        train_fn=train_fn,
        eval_fn=eval_fn,
        X_train=X_train,
        y_train=y_train,
        model_name=args.model,
        rep_name=args.rep,
        dataset_name=args.dataset,
        X_test=X_test,
        y_test=y_test,
    )

    wandb.log({"final_test": test_metrics})
    wandb.finish()


def run_polyatomic(args):
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    featurizer = REPRESENTATIONS[args.rep]

    def train_fn(X_tr, y_tr, log_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        dataset_cache_path = os.path.join(
            datasets_dir, f"polyatomic_data_{args.dataset}.pt"
        )
        if os.path.exists(dataset_cache_path):
            print(f"Loading cached dataset from {dataset_cache_path}")
            data_list = torch.load(dataset_cache_path, weights_only=False)
        else:
            assert os.path.exists(
                datasets_dir
            ), "Datasets directory still missing before featurizing."
            data_list = featurize_dataset_parallel(X_tr, y_tr, featurizer)
            torch.save(data_list, dataset_cache_path)

        model_cls = GNN_MODELS[args.model]
        loader = DataLoader(
            data_list, batch_size=32, shuffle=True, num_workers=8, pin_memory=False
        )

        node_feat_dim = data_list[0].x.shape[1]
        edge_feat_dim = data_list[0].edge_attr.shape[1]
        graph_feat_dim = data_list[0].graph_feats.shape[0]
        from torch_geometric.utils import degree

        deg = torch.zeros(32, dtype=torch.long)
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            bc = torch.bincount(d, minlength=deg.size(0))
            if bc.size(0) > deg.size(0):
                # Expand deg to fit
                new_deg = torch.zeros(bc.size(0), dtype=torch.long)
                new_deg[: deg.size(0)] = deg
                deg = new_deg
            deg += bc

        model = PolyatomicNet(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            graph_feat_dim=graph_feat_dim,
            deg=deg,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_gnn_model(model, loader, optimizer, log_file)
        return model

    def eval_fn(model, X_te, y_te, log_file, scaler, return_preds=False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        dataset_cache_path = os.path.join(
            datasets_dir, f"polyatomic_test_data_{args.dataset}.pt"
        )
        if os.path.exists(dataset_cache_path):
            print(f"Loading cached dataset from {dataset_cache_path}")
            data_list = torch.load(dataset_cache_path, weights_only=False)
        else:
            assert os.path.exists(
                datasets_dir
            ), "Datasets directory still missing before featurizing."
            data_list = featurize_dataset_parallel(X_te, y_te, featurizer)
            torch.save(data_list, dataset_cache_path)

        loader = DataLoader(data_list, batch_size=32)
        return eval_gnn_model(
            model, loader, log_file, scaler, return_preds=return_preds
        )

    wandb.init(
        project="chem-reps",
        name=f"GNN-{args.model}-{args.rep}-{args.dataset}",
        config=vars(args),
    )

    fold_metrics, test_metrics = k_fold_eval(
        train_fn=train_fn,
        eval_fn=eval_fn,
        X_train=X_train,
        y_train=y_train,
        model_name=args.model,
        rep_name=args.rep,
        dataset_name=args.dataset,
        X_test=X_test,
        y_test=y_test,
    )

    wandb.log({"final_test": test_metrics})
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gcn", "gin", "gat", "sage", "gp", "polyatomic"],
        required=True,
    )
    parser.add_argument("--rep", choices=list(REPRESENTATIONS.keys()), required=True)
    parser.add_argument(
        "--dataset",
        choices=[
            "esol",
            "freesolv",
            "lipophil",
            "boilingpoint",
            "qm9",
            "ic50",
            "bindingdb",
        ],
        required=True,
    )
    args = parser.parse_args()

    if args.model == "gp":
        run_gp(args)
    elif args.model != "polyatomic":
        run_gnn(args)
    else:
        run_polyatomic(args)


if __name__ == "__main__":
    main()
