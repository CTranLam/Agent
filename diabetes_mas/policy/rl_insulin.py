"""Q-learning dạng bảng đề xuất liều bù, trên simulator ĐỒ CHƠI. Chỉ đề xuất, rule an toàn quyết định.
python -m diabetes_mas.policy.rl_insulin [--eval]"""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import config as C

ACTIONS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
G_EDGES = np.arange(70, 310, 15)                  # 17 ô
TREND_EDGES = np.array([-6, -3, -1, 1, 3, 6])     # 7 ô
ISF_EDGES = np.array([20, 25, 30, 35, 40, 50, 60, 80])  # 9 ô
SHAPE = (len(G_EDGES) + 1, len(TREND_EDGES) + 1, len(ISF_EDGES) + 1, len(ACTIONS))
UNSEEN = -1e9


def encode(g, trend, isf):
    return (np.digitize(g, G_EDGES), np.digitize(trend, TREND_EDGES), np.digitize(isf, ISF_EDGES))


def reward_fn(g1):
    g1 = np.asarray(g1, dtype=float)
    r = -np.abs(g1 - C.TARGET_GLUCOSE) / 20.0 - 10.0 * (g1 < 70) - 20.0 * (g1 < 54) - 1.0 * (g1 > 180)
    return float(r) if r.ndim == 0 else r


class Simulator:
    """reset(n) rút n tình huống cùng nhiễu của chúng; step() tất định sau đó, nên so được hai liều trên cùng ca."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, n: int = 1):
        r = self.rng
        self.g, self.trend, self.isf = r.uniform(60, 350, n), r.uniform(-6, 6, n), r.uniform(20, 80, n)
        self.eff, self.noise = r.uniform(0.85, 1.15, n), r.normal(0, 8, n) + r.normal(0, 5, n)
        return self.g, self.trend, self.isf

    def step(self, units):
        g1 = self.g + 6 * self.trend - self.isf * np.asarray(units) * self.eff + self.noise
        return g1, reward_fn(g1)


class QPolicy:
    def __init__(self):
        self.Q = np.full(SHAPE, UNSEEN)
        self.Q[..., 0] = UNSEEN / 2  # ô chưa từng thấy: argmax về liều 0, không đoán

    def train(self, steps: int = 2_000_000, seed: int = 0):
        """gamma = 0, khám phá đều: Q(s,a) <- Q + (r-Q)/n hội tụ về trung bình phần thưởng, tính theo lô."""
        sim, rng = Simulator(seed), np.random.default_rng(seed + 1)
        sums, counts = np.zeros(SHAPE), np.zeros(SHAPE)
        for start in range(0, steps, 100_000):
            n = min(100_000, steps - start)
            g, t, isf = sim.reset(n)
            a = rng.integers(0, len(ACTIONS), n)
            _, r = sim.step(ACTIONS[a])
            idx = (*encode(g, t, isf), a)
            np.add.at(sums, idx, r)
            np.add.at(counts, idx, 1)
        seen = counts > 0
        self.Q[seen] = sums[seen] / counts[seen]
        return self

    def suggest(self, glucose: float, trend5: float, isf: float) -> float:
        return float(ACTIONS[int(np.argmax(self.Q[encode(glucose, trend5, isf)]))])

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
        return self


def _snap(units: float) -> float:
    return float(ACTIONS[ACTIONS <= max(units, 0)].max())


def plain_formula(g, trend, isf) -> float:
    return _snap((g - C.TARGET_GLUCOSE) / isf)


def trend_formula(g, trend, isf) -> float:
    return _snap((g + 6 * trend - C.TARGET_GLUCOSE) / isf)


def evaluate(policy_fn, n: int = 20000, seed: int = 1) -> dict:
    sim = Simulator(seed)
    g, t, isf = sim.reset(n)
    units = np.array([policy_fn(float(a), float(b), float(c)) for a, b, c in zip(g, t, isf)])
    g1, _ = sim.step(units)
    g0, _ = sim.step(np.zeros(n))
    return {"tir": float(((g1 >= 70) & (g1 <= 180)).mean()),
            "dose_hypo": float(((g1 < 70) & (g0 >= 70)).mean()),  # hạ do liều, không tính ca đã thấp sẵn
            "above_180": float((g1 > 180).mean())}


def compare(q: QPolicy) -> dict:
    return {"data": "simulator đồ chơi, 20.000 quyết định; chưa áp rule an toàn",
            "no_insulin": evaluate(lambda *a: 0.0), "formula": evaluate(plain_formula),
            "formula_trend": evaluate(trend_formula), "rl": evaluate(q.suggest)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    out = Path("artifacts")
    if args.eval:
        res = compare(QPolicy().load(out / "q_table.npy"))
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["rl_metrics"] = res
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        QPolicy().train().save(out / "q_table.npy")
        print("Đã lưu artifacts/q_table.npy")
