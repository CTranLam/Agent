"""Huấn luyện DiabetesNet: python -m diabetes_mas.dl.train"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .features import FEATURES, N_NUMERIC, encode_df, normalize
from .models import DiabetesNet


def split_dedup(df: pd.DataFrame, seed: int = 42):
    """Bỏ dòng trùng TRƯỚC khi chia để cùng một dòng không nằm ở cả train và test."""
    df = df.drop_duplicates()
    y = df["diabetes"]
    train, rest = train_test_split(df, test_size=0.30, random_state=seed, stratify=y)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed, stratify=rest["diabetes"])
    return train, val, test


def pick_threshold(y_true, probs, min_recall: float = 0.90) -> float:
    """Ngưỡng cao nhất vẫn đạt recall yêu cầu (sàng lọc: bỏ sót tệ hơn báo nhầm)."""
    pos = np.sort(np.asarray(probs)[np.asarray(y_true) == 1])[::-1]
    return float(pos[max(math.ceil(min_recall * len(pos)), 1) - 1])


def _metrics(y, p, thr) -> dict:
    pred = p >= thr
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}


def train(csv: str = "data/diabetes.csv", out: str = "artifacts", seed: int = 42, epochs: int = 50,
          patience: int = 5) -> dict:
    torch.manual_seed(seed)
    raw = pd.read_csv(csv)
    tr, va, te = split_dedup(raw, seed)
    Xtr_raw, Xva_raw, Xte_raw = encode_df(tr), encode_df(va), encode_df(te)
    mean, std = Xtr_raw[:, :N_NUMERIC].mean(0), Xtr_raw[:, :N_NUMERIC].std(0)
    Xtr, Xva, Xte = (normalize(x, mean, std) for x in (Xtr_raw, Xva_raw, Xte_raw))
    ytr, yva, yte = (d["diabetes"].to_numpy(np.float32) for d in (tr, va, te))

    model = DiabetesNet(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, yt, Xv, yv = map(torch.from_numpy, (Xtr, ytr, Xva, yva))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt)).split(512):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yv).item()
        print(f"epoch {ep + 1}: val_loss={v:.4f}")
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv)).numpy()
        pt = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()
    thr = pick_threshold(yva, pv, 0.90)

    logit = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    i_hba1c, i_glu = FEATURES.index("hba1c"), FEATURES.index("glucose")
    meta_new = {
        "mean": mean.tolist(), "std": std.tolist(), "threshold": thr,
        "n_rows_raw": len(raw), "n_rows_dedup": len(tr) + len(va) + len(te), "epochs_run": ep + 1,
        "metrics": _metrics(yte, pt, thr),
        "baselines": {"hba1c_only_roc_auc": roc_auc_score(yte, Xte_raw[:, i_hba1c]),
                      "glucose_only_roc_auc": roc_auc_score(yte, Xte_raw[:, i_glu]),
                      "logistic_roc_auc": roc_auc_score(yte, logit.predict_proba(Xte)[:, 1])},
    }
    outp = Path(out)
    outp.mkdir(exist_ok=True)
    torch.save(model.state_dict(), outp / "diabetes_net.pt")
    meta_path = outp / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(meta_new)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_new


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
