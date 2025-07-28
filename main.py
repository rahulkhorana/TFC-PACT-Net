import argparse
import wandb
import numpy as np
from data.loaders import load_dataset
from data.featurize import (
    smiles_to_graph,
    selfies_to_graph,
    ecfp_to_graph,
    smiles_for_gp,
    selfies_for_gp,
    ecfp_for_gp,
)
from models.gnn import GCN, GIN, GAT, GraphSAGE
from models.gp import TanimotoGP as GPModel
from training.train_eval import (
    train_gp_model,
    eval_gp_model,
    train_gnn_model,
    eval_gnn_model,
    k_fold_eval,
)
import torch
from torch_geometric.data import DataLoader

# Representation functions for GNN
REPRESENTATIONS = {
    "smiles": smiles_to_graph,
    "selfies": selfies_to_graph,
    "ecfp": ecfp_to_graph,
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
}


def featurize_dataset(X, y, featurizer):
    data_list = []
    for xi, yi in zip(X, y):
        g = featurizer(xi)
        g.y = torch.tensor([yi], dtype=torch.float)
        data_list.append(g)
    return data_list


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
        return eval_gnn_model(model, loader, log_file, scaler)

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
        "--model", choices=["gcn", "gin", "gat", "sage", "gp"], required=True
    )
    parser.add_argument("--rep", choices=list(REPRESENTATIONS.keys()), required=True)
    parser.add_argument(
        "--dataset", choices=["esol", "freesolv", "lipophil"], required=True
    )
    args = parser.parse_args()

    if args.model == "gp":
        run_gp(args)
    else:
        run_gnn(args)


if __name__ == "__main__":
    main()
