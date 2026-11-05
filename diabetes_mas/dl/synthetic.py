"""CGM MÔ PHỎNG (dataset có sẵn không có chuỗi đường huyết). Mỗi điểm cách nhau 5 phút."""
import numpy as np


def _impulse(peak: float, length: int) -> np.ndarray:
    t = np.arange(length) / peak
    return t * np.exp(1 - t)  # đạt 1 tại t=peak rồi giảm dần


def simulate_cgm(n_series: int, length: int = 288, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    meal_h, ins_h = _impulse(10, 72), _impulse(16, 96)
    out = np.empty((n_series, length))
    for i in range(n_series):
        y = np.full(length + 96, rng.uniform(100, 145))
        t = int(rng.integers(0, 40))
        while t < length:
            amp = rng.uniform(30, 110)
            n = len(meal_h)
            y[t:t + n] += amp * meal_h[:len(y) - t][:n]
            # liều không khớp bữa ăn: nguồn tạo hạ đường huyết
            ins_amp = amp * rng.uniform(0.4, 1.7)
            s = t + int(rng.integers(0, 5))
            n = len(ins_h)
            y[s:s + n] -= ins_amp * ins_h[:len(y) - s][:n]
            t += int(rng.integers(45, 90))
        drift = np.zeros(len(y))
        for k in range(1, len(y)):
            drift[k] = 0.97 * drift[k - 1] + rng.normal(0, 0.8)
        out[i] = np.clip((y + drift)[:length] + rng.normal(0, 1.0, length), 40, 400)
    return out
