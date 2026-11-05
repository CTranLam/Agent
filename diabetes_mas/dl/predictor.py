"""Mặt tiền cho các agent: classify(profile) và forecast(readings)."""
import json
from pathlib import Path

import numpy as np
import torch

from .decision import classify_with_prob, conservative_forecast, pad_window
from .features import FEATURES, encode_profile, normalize
from .models import DiabetesNet, GlucoseLSTM
from .train_lstm import G_MEAN, G_STD


class Predictor:
    def __init__(self, artifacts="artifacts"):
        a = Path(artifacts)
        if not all((a / f).exists() for f in ("meta.json", "diabetes_net.pt", "lstm.pt")):
            raise RuntimeError(f"Thiếu model trong {a}. Hãy chạy: python -m diabetes_mas.dl.train "
                               "và python -m diabetes_mas.dl.train_lstm")
        self.meta = json.loads((a / "meta.json").read_text())
        self.net = DiabetesNet(len(FEATURES))
        self.net.load_state_dict(torch.load(a / "diabetes_net.pt", weights_only=True))
        self.net.eval()
        self.lstm = GlucoseLSTM()
        self.lstm.load_state_dict(torch.load(a / "lstm.pt", weights_only=True))
        self.lstm.eval()

    def classify(self, profile) -> dict:
        # thiếu đường huyết đói: ước lượng từ HbA1c (ADAG) chỉ làm đầu vào mạng, không dùng chẩn đoán
        g = profile.fasting_glucose if profile.fasting_glucose is not None else 28.7 * profile.hba1c - 46.7
        x = normalize(encode_profile(profile, g), self.meta["mean"], self.meta["std"])
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.from_numpy(x[None]))).item()
        return classify_with_prob(profile, p, self.meta["threshold"])

    def forecast(self, readings) -> dict:
        w = pad_window(readings)
        x = torch.tensor([[[(v - G_MEAN) / G_STD] for v in w]], dtype=torch.float32)
        with torch.no_grad():
            delta = self.lstm(x)[0].numpy()
        return conservative_forecast(w, np.asarray(w[-1]) + delta * G_STD, len(readings))
