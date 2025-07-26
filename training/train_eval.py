import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import logging
import os
from datetime import datetime

# Raw text logging setup
LOG_ROOT = "logs"
os.makedirs(LOG_ROOT, exist_ok=True)


def setup_log_file(model_name, rep_name, dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{model_name}_{rep_name}_{dataset_name}_{timestamp}.txt"
    path = os.path.join(LOG_ROOT, fname)
    return open(path, "w")


def write_log(log_file, text):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def train_gnn_model(model, loader, optimizer, log_file, loss_fn=torch.nn.MSELoss()):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training GNN"):
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()
        out = model(batch).squeeze()
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    write_log(log_file, f"GNN Train Loss: {total_loss / len(loader):.4f}")
    return total_loss / len(loader)


def eval_gnn_model(model, loader, log_file, scaler):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating GNN"):
            batch = batch.to(next(model.parameters()).device)
            out = model(batch).squeeze()
            y_true.append(batch.y.cpu())
            y_pred.append(out.cpu())
    y_true = scaler.inverse_transform(torch.cat(y_true).numpy().reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(torch.cat(y_pred).numpy().reshape(-1, 1)).ravel()
    return report_metrics(y_true, y_pred, log_file)


def train_gp_model(gp_model, X_train, y_train, log_file):
    write_log(log_file, "Training GP model...")
    gp_model.fit(X_train, y_train)
    return gp_model


def eval_gp_model(gp_model, X_test, y_test, log_file, scaler):
    write_log(log_file, "Evaluating GP model...")
    y_pred, _ = gp_model.predict(X_test)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    return report_metrics(y_test, y_pred, log_file)


def report_metrics(y_true, y_pred, log_file):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    write_log(log_file, f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def bootstrap_ci(arr, n_boot=1000, ci=95):
    boot_means = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(boot_means), (lower, upper)


def k_fold_eval(
    train_fn, eval_fn, data, targets, model_name, rep_name, dataset_name, k=5, seed=42
):
    log_file = setup_log_file(model_name, rep_name, dataset_name)
    write_log(log_file, f"Experiment: {model_name} + {rep_name} on {dataset_name}")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        write_log(log_file, f"\nFOLD {fold+1}/{k}")
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()

        model = train_fn(X_train, y_train_scaled, log_file)
        metrics = eval_fn(model, X_test, y_test_scaled, log_file, scaler)
        results.append(metrics)

    write_log(log_file, "\n================ FINAL RESULTS ================")
    metrics_keys = results[0].keys()
    for key in metrics_keys:
        metric_vals = [m[key] for m in results]
        mean, (low, high) = bootstrap_ci(metric_vals)
        write_log(
            log_file,
            f"{key.upper()}: {mean:.4f} [{low:.4f}, {high:.4f}] (95% bootstrap CI)",
        )

    log_file.close()
    return results
