import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
from pathlib import Path

# polyatomic
from torch.amp.autocast_mode import autocast

ROOT = Path(__file__).parent.parent.resolve().__str__()
LOG_ROOT = Path(ROOT + "/" + "logs_hyperparameter")
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT, exist_ok=False)


def setup_log_file(model_name, rep_name, dataset_name):
    from pathlib import Path
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{model_name}_{rep_name}_{dataset_name}_{timestamp}.txt"
    parent = Path(__file__).parent.parent.resolve().__str__()
    log_dir = Path(parent + "/" + "logs_hyperparameter")
    if not os.path.exists(log_dir):
        os.makedirs(LOG_ROOT, exist_ok=False)

    log_path = log_dir / fname
    print(f"[Logging] Writing to: {log_path}")
    return open(log_path, "w")


def write_log(log_file, text):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def train_gnn_model(model, loader, optimizer, log_file, loss_fn=torch.nn.MSELoss()):
    total_loss = 0
    for _ in range(20):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            optimizer.zero_grad()
            out = model(batch).squeeze()
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    write_log(log_file, f"GNN Train Loss: {avg_loss:.4f}")
    return avg_loss


def eval_gnn_model(model, loader, log_file, scaler, return_preds=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating GNN"):
            batch = batch.to(next(model.parameters()).device)
            out = model(batch).view(-1)
            y_true.append(batch.y.cpu())
            y_pred.append(out.cpu())
    y_true = scaler.inverse_transform(torch.cat(y_true).numpy().reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(torch.cat(y_pred).numpy().reshape(-1, 1)).ravel()
    metrics = report_metrics(y_true, y_pred, log_file)
    if return_preds:
        return metrics, y_true, y_pred
    return metrics


def train_gp_model(gp_model, X_train, y_train, log_file):
    write_log(log_file, "Training GP model...")
    gp_model.fit(X_train, y_train)
    return gp_model


def eval_gp_model(gp_model, X_test, y_test, log_file, scaler, return_preds=False):
    write_log(log_file, "Evaluating GP model...")
    y_pred, _ = gp_model.predict(X_test)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    metrics = report_metrics(y_test, y_pred, log_file)
    if return_preds:
        return metrics, y_test, y_pred
    return metrics


def report_metrics(y_true, y_pred, log_file):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
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


def bootstrap_metric_ci(metric_fn, y_true, y_pred, n_boot=1000, ci=95, rng=None):
    rng = np.random.default_rng(rng)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_vals.append(metric_fn(y_true[idx], y_pred[idx]))
    mean_val = np.mean(boot_vals)
    lo, hi = np.percentile(boot_vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return mean_val, (lo, hi)


def train_polyatomic(
    model, loader, optimizer, loss_fn, scaler_grad, device, scheduler, accum_steps=1
):
    """
    custom training loop for polyatomic GNNs
    uses mixed precision training with autocast
    accum_steps allows gradient accumulation for larger effective batch size
    this was designed for GPU training, but here is in CPU mode
    """
    use_amp = torch.cuda.is_available()
    for _ in range(20):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            batch = batch.to(device)
            batch.x = batch.x.float()
            batch.edge_attr = batch.edge_attr.float()
            batch.graph_feats = batch.graph_feats.float()
            batch.y = batch.y.float()

            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    output = model(batch)
                    loss = loss_fn(output, batch.y.view(-1)) / accum_steps
            else:
                output = model(batch)
                loss = loss_fn(output, batch.y.view(-1)) / accum_steps

            if use_amp:
                scaler_grad.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (i + 1) % accum_steps == 0 or (i + 1 == len(loader)):
                if use_amp:
                    scaler_grad.step(optimizer)
                    scaler_grad.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            total_loss += loss.item() * batch.num_graphs * accum_steps

        avg_loss = total_loss / len(loader.dataset)
        if scheduler is not None:
            scheduler.step(avg_loss)

    return model


def evaluate_polyatomic(model, loader, device, log_file, scaler, return_preds=False):
    """
    uses autocast for mixed precision evaluation
    this was designed for GPU training, but here is in CPU mode
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad(), autocast(
        device_type="cpu", dtype=torch.float16
    ):  # change to 'cuda' if using GPU
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.view(-1))
            trues.append(batch.y.view(-1))
    y_pred = torch.cat(preds)
    y_test = torch.cat(trues)
    y_pred = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1)).ravel()
    y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).ravel()
    metrics = report_metrics(y_test, y_pred, log_file)
    if return_preds:
        return metrics, y_test, y_pred
    return metrics


def k_fold_eval(
    train_fn,
    eval_fn,
    X_train,
    y_train,
    model_name,
    rep_name,
    dataset_name,
    X_test,
    y_test,
    k=5,
    seed=42,
    log_file=None,
):
    log_file = setup_log_file(model_name, rep_name, dataset_name)
    write_log(log_file, f"Experiment: {model_name}+{rep_name} on {dataset_name}")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics, fold_models = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        write_log(log_file, f"\nFOLD {fold+1}/{k}")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        scaler = StandardScaler()
        y_tr_s = scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_val_s = scaler.transform(y_val.reshape(-1, 1)).ravel()

        model = train_fn(X_tr, y_tr_s, log_file)
        m = eval_fn(model, X_val, y_val_s, log_file, scaler)
        fold_metrics.append(m)
        fold_models.append((model, m["rmse"]))

    write_log(log_file, "\n====== K-FOLD SUMMARY ======")
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        mean, (lo, hi) = bootstrap_ci(vals)
        write_log(log_file, f"{key.upper()}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    best_idx = np.argmin([rm for (_, rm) in fold_models])
    best_model = fold_models[best_idx][0]
    write_log(log_file, f"\n★ Using fold {best_idx+1} model for test inference")

    test_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_test_s = test_scaler.transform(y_test.reshape(-1, 1)).ravel()
    test_metrics, y_true_test, y_pred_test = eval_fn(
        best_model, X_test, y_test_s, log_file, test_scaler, return_preds=True
    )

    write_log(log_file, f"\n====== HELD-OUT TEST METRICS ======\n{test_metrics}")
    rmse_mean, (rmse_lo, rmse_hi) = bootstrap_metric_ci(
        lambda a, b: np.sqrt(mean_squared_error(a, b)), y_true_test, y_pred_test
    )
    mae_mean, (mae_lo, mae_hi) = bootstrap_metric_ci(
        mean_absolute_error, y_true_test, y_pred_test
    )
    r2_mean, (r2_lo, r2_hi) = bootstrap_metric_ci(r2_score, y_true_test, y_pred_test)
    write_log(
        log_file,
        f"Test RMSE: {rmse_mean:.4f}  (95 % CI: {rmse_lo:.4f}–{rmse_hi:.4f})\n",
    )
    write_log(
        log_file,
        f"Test MAE : {mae_mean :.4f}  (95 % CI: {mae_lo :.4f}–{mae_hi :.4f})\n",
    )
    write_log(
        log_file,
        f"Test R²  : {r2_mean  :.4f}  (95 % CI: {r2_lo  :.4f}–{r2_hi  :.4f})\n",
    )
    log_file.close()
    return fold_metrics, test_metrics
