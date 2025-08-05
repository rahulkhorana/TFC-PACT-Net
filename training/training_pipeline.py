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

ROOT = Path(__file__).parent.parent.resolve().__str__()
LOG_ROOT = Path(ROOT + "/" + "logs_hyperparameter")
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT, exist_ok=False)


def setup_log_file(args):
    from pathlib import Path
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name, rep_name, dataset_name = args.model, args.rep, args.dataset
    fname = f"{model_name}_{rep_name}_{dataset_name}_{timestamp}.txt"
    parent = Path(__file__).parent.parent.resolve().__str__()
    log_dir = Path(parent + "/" + "logs_hyperparameter" + "/" + f"{args.dataset}")
    if not os.path.exists(log_dir):
        os.makedirs(LOG_ROOT, exist_ok=False)

    log_path = log_dir / fname
    print(f"[Logging] Writing to: {log_path}")
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


def bootstrap_metric(y_true, y_pred, metric_func, n_bootstraps=1000):
    """Performs bootstrapping to estimate the confidence interval of a metric."""
    n_samples = len(y_true)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    lower_bound = np.percentile(bootstrapped_scores, 2.5)
    upper_bound = np.percentile(bootstrapped_scores, 97.5)
    mean_score = np.mean(bootstrapped_scores)

    return mean_score, lower_bound, upper_bound


def rmse_func(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_func(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def k_fold_tuned_eval(args, train_graphs_full, test_graphs):
    """
    Orchestrates a rigorous NESTED cross-validation workflow.
    1. Outer Loop (Evaluation): Splits data to get an unbiased performance estimate.
    2. Inner Loop (Tuning): Finds the best hyperparameters for each outer fold's training set.
    3. Final Model: After getting the performance estimate, finds the best params on the full
       training set and evaluates on the held-out test set.
    """
    log_file_path = setup_log_file(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Nested Cross-Validation for Unbiased Performance Estimate ---
    write_log(
        log_file_path,
        "\n===== STEP 1: Nested Cross-Validation Performance Estimation =====",
    )
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_fold_rmses = []
    val_fold_maes = []

    # Use indices for splitting
    train_indices = np.arange(len(train_graphs_full))

    for fold, (train_idx, val_idx) in enumerate(outer_kf.split(train_indices)):
        write_log(log_file_path, f"\n--- OUTER FOLD {fold + 1}/5 ---")

        # Create data subsets for this outer fold
        train_fold_graphs = [train_graphs_full[i] for i in train_idx]
        val_fold_graphs = [train_graphs_full[i] for i in val_idx]

        # --- Inner Loop: Hyperparameter Tuning on this fold's training data ---
        write_log(
            log_file_path,
            "INFO: Finding best hyperparameters for this fold (Inner Loop)...",
        )
        # The HPO function is now called on the training subset of the outer fold
        best_params_for_fold = find_best_hyperparameters(
            args, train_fold_graphs, device
        )
        write_log(
            log_file_path,
            f"INFO: Best params for fold {fold + 1}: {best_params_for_fold}",
        )

        # --- Train and Evaluate this fold's model ---
        # Fit scaler ONLY on the training data for this fold
        y_train_fold_raw = np.array([g.y.item() for g in train_fold_graphs]).reshape(
            -1, 1
        )
        scaler = StandardScaler().fit(y_train_fold_raw)

        # Apply scaling (on copies to not affect other folds)
        train_fold_graphs_scaled = [g.clone() for g in train_fold_graphs]
        val_fold_graphs_scaled = [g.clone() for g in val_fold_graphs]
        for g in train_fold_graphs_scaled:
            g.y = torch.tensor(scaler.transform(g.y.reshape(1, -1)), dtype=torch.float)
        for g in val_fold_graphs_scaled:
            g.y = torch.tensor(scaler.transform(g.y.reshape(1, -1)), dtype=torch.float)

        # Create DataLoaders
        train_loader = DataLoader(
            train_fold_graphs_scaled,
            batch_size=best_params_for_fold["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_fold_graphs_scaled, batch_size=best_params_for_fold["batch_size"]
        )

        # Train model for this fold
        model = get_model_instance(args, best_params_for_fold, train_fold_graphs).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params_for_fold["lr"])
        train_gnn_model(model, train_loader, val_loader, optimizer, device)

        # Evaluate on the validation set for this fold
        y_true_val, y_pred_val = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch).view(-1)
                y_true_val.extend(
                    scaler.inverse_transform(
                        batch.y.cpu().numpy().reshape(-1, 1)
                    ).ravel()
                )
                y_pred_val.extend(
                    scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).ravel()
                )

        fold_rmse = rmse_func(np.array(y_true_val), np.array(y_pred_val))
        fold_mae = mae_func(np.array(y_true_val), np.array(y_pred_val))
        val_fold_rmses.append(fold_rmse)
        val_fold_maes.append(fold_mae)
        write_log(
            log_file_path,
            f"INFO: Fold {fold + 1} Val RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}",
        )

    # --- Step 2: Report overall validation performance from Nested CV ---
    mean_val_rmse = np.mean(val_fold_rmses)
    std_val_rmse = np.std(val_fold_rmses)
    mean_val_mae = np.mean(val_fold_maes)
    std_val_mae = np.std(val_fold_maes)
    write_log(log_file_path, "\n------ Nested Cross-Validation Summary ------")
    write_log(
        log_file_path,
        f"Unbiased Validation RMSE: {mean_val_rmse:.4f} ± {std_val_rmse:.4f}",
    )
    write_log(
        log_file_path,
        f"Unbiased Validation MAE:  {mean_val_mae:.4f} ± {std_val_mae:.4f}",
    )
    write_log(log_file_path, f"VAL FOLD RMSEs: {val_fold_rmses}")
    write_log(log_file_path, f"VAL FOLD MAEs: {val_fold_maes}")

    # --- Step 3: Train the final model for deployment/testing ---
    write_log(log_file_path, "\n===== STEP 2: Final Model Training & Testing =====")
    write_log(
        log_file_path,
        "INFO: Finding best hyperparameters on the FULL train/val set for final model...",
    )
    final_best_params = find_best_hyperparameters(args, train_graphs_full, device)
    write_log(
        log_file_path,
        f"INFO: Optimal hyperparameters for final model: {final_best_params}",
    )

    write_log(log_file_path, "INFO: Training final model...")
    y_train_full_raw = np.array([g.y.item() for g in train_graphs_full]).reshape(-1, 1)
    final_scaler = StandardScaler().fit(y_train_full_raw)

    final_train_graphs = [g.clone() for g in train_graphs_full]
    for g in final_train_graphs:
        g.y = torch.tensor(
            final_scaler.transform(g.y.reshape(1, -1)), dtype=torch.float
        )

    train_subset, val_subset = train_test_split(
        final_train_graphs, test_size=0.1, random_state=42
    )
    final_train_loader = DataLoader(
        train_subset, batch_size=final_best_params["batch_size"], shuffle=True
    )
    final_val_loader = DataLoader(
        val_subset, batch_size=final_best_params["batch_size"]
    )

    final_model = get_model_instance(args, final_best_params, final_train_graphs).to(
        device
    )
    final_optimizer = torch.optim.Adam(
        final_model.parameters(), lr=final_best_params["lr"]
    )
    train_gnn_model(
        final_model, final_train_loader, final_val_loader, final_optimizer, device
    )

    # --- Step 4: Evaluate the final model on the held-out test set ---
    write_log(log_file_path, "\n===== STEP 3: Final Held-Out Test Evaluation =====")
    final_test_graphs = [g.clone() for g in test_graphs]
    for g in final_test_graphs:
        g.y = torch.tensor(
            final_scaler.transform(g.y.reshape(1, -1)), dtype=torch.float
        )
    test_loader = DataLoader(
        final_test_graphs, batch_size=final_best_params["batch_size"]
    )

    y_true_test, y_pred_test = [], []
    final_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = final_model(batch).view(-1)
            y_true_test.extend(
                final_scaler.inverse_transform(
                    batch.y.cpu().numpy().reshape(-1, 1)
                ).ravel()
            )
            y_pred_test.extend(
                final_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).ravel()
            )

    y_true_test, y_pred_test = np.array(y_true_test), np.array(y_pred_test)

    rmse_mean, rmse_low, rmse_high = bootstrap_metric(
        y_true_test, y_pred_test, rmse_func
    )
    mae_mean, mae_low, mae_high = bootstrap_metric(y_true_test, y_pred_test, mae_func)

    write_log(
        log_file_path,
        f"Test RMSE: {rmse_mean:.4f} (95% CI: [{rmse_low:.4f}, {rmse_high:.4f}])",
    )
    write_log(
        log_file_path,
        f"Test MAE:  {mae_mean:.4f} (95% CI: [{mae_low:.4f}, {mae_high:.4f}])",
    )

    results_data = {
        "val_rmse_mean": [mean_val_rmse],
        "val_rmse_std": [std_val_rmse],
        "val_mae_mean": [mean_val_mae],
        "val_mae_std": [std_val_mae],
        "test_rmse_mean": [rmse_mean],
        "test_rmse_ci_low": [rmse_low],
        "test_rmse_ci_high": [rmse_high],
        "test_mae_mean": [mae_mean],
        "test_mae_ci_low": [mae_low],
        "test_mae_ci_high": [mae_high],
    }
    parent = Path(__file__).parent.parent.resolve().__str__()
    log_dir = Path(
        parent + "/" + "logs_hyperparameter" + "/" + f"{args.dataset}"
    ).__str__()
    data_res_path = (
        log_dir + "/" + f"{args.model}_{args.rep}_{args.dataset}_final_results.csv"
    )
    return pd.DataFrame(results_data).to_csv(data_res_path)
