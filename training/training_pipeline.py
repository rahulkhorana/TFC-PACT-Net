import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import tempfile
from pathlib import Path
from datetime import datetime
import optuna
import pandas as pd
from torch_geometric.loader import DataLoader

# Import project modules
from data.data_handling import get_model_instance
from plotting import plot_parity


def setup_log_file(args):
    """Sets up a unique log file for an experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"TUNED_{args.model}_{args.rep}_{args.dataset}_{timestamp}.txt"
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / fname
    print(f"[Logging] Writing to: {log_path}")
    with open(log_path, "w") as f:
        f.write(f"Experiment Config: {vars(args)}\n\n")
    return log_path


def write_log(log_file_path, text):
    """Writes a message to both console and the log file."""
    print(text)
    with open(log_file_path, "a") as f:
        f.write(text + "\n")


def train_gnn_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    loss_fn=nn.L1Loss(),
    max_epochs=200,
    patience=20,
):
    """Trains a GNN with early stopping based on a validation set."""
    best_val_loss = float("inf")
    epochs_no_improve = 0
    temp_dir = tempfile.gettempdir()
    best_model_path = os.path.join(temp_dir, f"best_model_{os.getpid()}.pt")

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += loss_fn(model(batch).view(-1), batch.y).item()
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

    return model


def objective(trial, args, train_graphs, val_graphs, device):
    """Optuna objective function. Now very fast as it uses pre-featurized data."""
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
    }

    y_for_scaler = np.array([g.y.item() for g in train_graphs])
    scaler = StandardScaler().fit(y_for_scaler.reshape(-1, 1))

    # Scale y-values in graph objects for this trial
    y_train_s = scaler.transform(
        np.array([g.y.item() for g in train_graphs]).reshape(-1, 1)
    ).ravel()
    for g, y_s in zip(train_graphs, y_train_s):
        g.y = torch.tensor([y_s], dtype=torch.float)

    y_val_s = scaler.transform(
        np.array([g.y.item() for g in val_graphs]).reshape(-1, 1)
    ).ravel()
    for g, y_s in zip(val_graphs, y_val_s):
        g.y = torch.tensor([y_s], dtype=torch.float)

    train_loader = DataLoader(
        train_graphs, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_graphs, batch_size=params["batch_size"])

    model = get_model_instance(args, params, train_graphs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    train_gnn_model(
        model, train_loader, val_loader, optimizer, device, loss_fn=nn.L1Loss()
    )

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            val_loss += nn.L1Loss()(model(batch).view(-1), batch.y).item()

    return val_loss / len(val_loader)


def find_best_hyperparameters(args, train_val_graphs, device):
    """Runs an Optuna study on a given list of pre-featurized graphs."""
    train_graphs, val_graphs = train_test_split(
        train_val_graphs, test_size=0.2, random_state=42
    )

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, args, train_graphs, val_graphs, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    return study.best_params


def k_fold_tuned_eval(args, train_graphs_full, test_graphs):
    """Orchestrates K-fold CV with per-fold hyperparameter tuning using pre-featurized data."""
    log_file_path = setup_log_file(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # This is the correct way to handle splitting a list for the PyG DataLoader
    train_indices = np.arange(len(train_graphs_full))

    for fold, (train_val_idx, _) in enumerate(kf.split(train_indices)):
        write_log(log_file_path, f"\n===== FOLD {fold + 1}/5 =====")

        # Use list comprehensions on indices to create subsets
        train_val_graphs = [train_graphs_full[i] for i in train_val_idx]

        write_log(log_file_path, "INFO: Finding best hyperparameters for this fold...")
        best_params = find_best_hyperparameters(args, train_val_graphs, device)
        write_log(
            log_file_path, f"INFO: Best params for fold {fold + 1}: {best_params}"
        )

        write_log(
            log_file_path,
            "INFO: Training final model for this fold with best params...",
        )
        y_for_scaler = np.array([g.y.item() for g in train_val_graphs])
        scaler = StandardScaler().fit(y_for_scaler.reshape(-1, 1))

        final_train_graphs, final_val_graphs = train_test_split(
            train_val_graphs, test_size=0.1, random_state=42
        )

        # Scale y-values for final training
        y_train_s = scaler.transform(
            np.array([g.y.item() for g in final_train_graphs]).reshape(-1, 1)
        ).ravel()
        for g, y_s in zip(final_train_graphs, y_train_s):
            g.y = torch.tensor([y_s], dtype=torch.float)

        y_val_s = scaler.transform(
            np.array([g.y.item() for g in final_val_graphs]).reshape(-1, 1)
        ).ravel()
        for g, y_s in zip(final_val_graphs, y_val_s):
            g.y = torch.tensor([y_s], dtype=torch.float)

        final_train_loader = DataLoader(
            final_train_graphs, batch_size=best_params["batch_size"], shuffle=True
        )
        final_val_loader = DataLoader(
            final_val_graphs, batch_size=best_params["batch_size"]
        )

        model = get_model_instance(args, best_params, train_val_graphs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

        train_gnn_model(model, final_train_loader, final_val_loader, optimizer, device)

        fold_results.append(
            {"model_state": model.state_dict(), "scaler": scaler, "params": best_params}
        )

    # Find the best model across all folds (based on which fold had the best HPO result, implicitly)
    # We will use the model from the last fold for simplicity here.
    best_fold = fold_results[-1]
    write_log(
        log_file_path, f"\n★ Using model from last fold for final test evaluation."
    )

    final_model = get_model_instance(args, best_fold["params"], train_graphs_full).to(
        device
    )
    final_model.load_state_dict(best_fold["model_state"])

    # Scale test data y-values
    y_test_s = (
        best_fold["scaler"]
        .transform(np.array([g.y.item() for g in test_graphs]).reshape(-1, 1))
        .ravel()
    )
    for g, y_s in zip(test_graphs, y_test_s):
        g.y = torch.tensor([y_s], dtype=torch.float)
    test_loader = DataLoader(test_graphs, batch_size=best_fold["params"]["batch_size"])

    y_true_test, y_pred_test = [], []
    final_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = final_model(batch).view(-1)
            y_true_test.extend(
                best_fold["scaler"]
                .inverse_transform(batch.y.cpu().numpy().reshape(-1, 1))
                .ravel()
            )
            y_pred_test.extend(
                best_fold["scaler"]
                .inverse_transform(out.cpu().numpy().reshape(-1, 1))
                .ravel()
            )

    y_true_test, y_pred_test = np.array(y_true_test), np.array(y_pred_test)
    rmse = np.sqrt(np.mean((y_true_test - y_pred_test) ** 2))
    mae = np.mean(np.abs(y_true_test - y_pred_test))

    write_log(
        log_file_path,
        f"\n====== FINAL HELD-OUT TEST METRICS ======\nRMSE: {rmse:.4f}\nMAE:  {mae:.4f}",
    )

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plot_filename = f"{args.model}_{args.rep}_{args.dataset}_parity.png"
    plot_parity(
        y_true_test,
        y_pred_test,
        f"Parity Plot for {args.model} on {args.dataset}",
        plot_dir / plot_filename,
    )

    results_data = {"final_test_rmse": [rmse], "final_test_mae": [mae]}
    return pd.DataFrame(results_data)
