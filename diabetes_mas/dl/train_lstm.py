"""Huấn luyện LSTM trên CGM mô phỏng: python -m diabetes_mas.dl.train_lstm"""
import json
from pathlib import Path

import numpy as np
import torch

from .models import GlucoseLSTM
from .synthetic import simulate_cgm

G_MEAN, G_STD = 130.0, 50.0


def split_by_series(series: np.ndarray, seed: int = 0):
    """Chia theo chuỗi TRƯỚC khi cắt cửa sổ; cắt rồi chia ngẫu nhiên sẽ rò rỉ dữ liệu."""
    idx = np.random.default_rng(seed).permutation(len(series))
    a, b = int(len(idx) * 0.7), int(len(idx) * 0.85)
    return series[idx[:a]], series[idx[a:b]], series[idx[b:]]


def windows(series: np.ndarray, in_len: int = 12, out_len: int = 6, stride: int = 1):
    X, Y = [], []
    for s in series:
        for i in range(0, len(s) - in_len - out_len + 1, stride):
            X.append(s[i:i + in_len])
            Y.append(s[i + in_len:i + in_len + out_len])
    return np.array(X, np.float32)[..., None], np.array(Y, np.float32)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """hypo_recall: trong các cửa sổ thực tế có <70, tỉ lệ dự báo cũng có <70."""
    real, pred = y_true.min(1) < 70, y_pred.min(1) < 70
    return {"mae": float(np.abs(y_true - y_pred).mean()),
            "hypo_recall": float(pred[real].mean()) if real.any() else None,
            "hypo_precision": float(real[pred].mean()) if pred.any() else None}


def linear_baseline(X: np.ndarray, horizon: int = 6) -> np.ndarray:
    w = X[..., 0][:, -4:]
    slope = ((w - w.mean(1, keepdims=True)) * (np.arange(4) - 1.5)).sum(1) / 5.0
    return w[:, -1:] + slope[:, None] * np.arange(1, horizon + 1)


def predict(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        delta = model(torch.from_numpy((X - G_MEAN) / G_STD)).numpy()
    return X[:, -1, 0:1] + delta * G_STD  # mô hình học phần thay đổi so với điểm cuối


def train_lstm(out: str = "artifacts", n_series: int = 300, epochs: int = 15, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    tr, va, te = split_by_series(simulate_cgm(n_series, seed=seed), seed)
    Xtr, Ytr = windows(tr, stride=3)
    Xva, Yva = windows(va, stride=3)
    Xte, Yte = windows(te)
    xt = torch.from_numpy((Xtr - G_MEAN) / G_STD)
    yt = torch.from_numpy((Ytr - Xtr[:, -1, 0:1]) / G_STD)
    model = GlucoseLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt)).split(256):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(xt[idx]), yt[idx]).backward()
            opt.step()
        v = forecast_metrics(Yva, predict(model, Xva))["mae"]
        print(f"epoch {ep + 1}: val_mae={v:.2f}", flush=True)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    # chỉ xét hạ đường huyết SẮP xảy ra: điểm hiện tại còn >=70
    upcoming = Xte[:, -1, 0] >= 70
    X, Y = Xte[upcoming], Yte[upcoming]
    hold = np.repeat(X[:, -1, 0:1], 6, axis=1)
    linear, lstm = linear_baseline(X), predict(model, X)
    result = {"data": "CGM mô phỏng, chia theo chuỗi",
              "n_test_series": len(te), "n_windows": len(X), "n_hypo_windows": int((Y.min(1) < 70).sum()),
              "hold_last": forecast_metrics(Y, hold), "linear": forecast_metrics(Y, linear),
              "lstm": forecast_metrics(Y, lstm), "conservative_min": forecast_metrics(Y, np.minimum(lstm, linear))}
    outp = Path(out)
    outp.mkdir(exist_ok=True)
    torch.save(model.state_dict(), outp / "lstm.pt")
    meta_path = outp / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["forecast_metrics"] = result
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    print(json.dumps(train_lstm(), indent=2, ensure_ascii=False))
