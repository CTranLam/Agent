"""Predictor giả dùng hàm quyết định thật, để test logic agent mà không cần model."""
import numpy as np

from diabetes_mas.dl.decision import classify_with_prob, conservative_forecast, pad_window


class FakePredictor:
    def __init__(self, prob: float = 0.01, thr: float = 0.09):
        self.prob, self.thr = prob, thr

    def classify(self, profile):
        return classify_with_prob(profile, self.prob, self.thr)

    def forecast(self, readings):
        w = pad_window(readings)
        slope = float(np.polyfit(np.arange(4), w[-4:], 1)[0])
        linear = [w[-1] + slope * (i + 1) for i in range(6)]
        return conservative_forecast(w, linear, len(readings))
