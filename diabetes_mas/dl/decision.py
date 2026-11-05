"""Từ xác suất đến kết luận: hàm thuần, không cần PyTorch."""
import math

import numpy as np

from ..schemas import PatientProfile

CLASS_NAMES = {"NONE": "Không tiểu đường", "PRE": "Tiền tiểu đường",
               "T1": "Tiểu đường type 1", "T2": "Tiểu đường type 2"}


def meets_ada_diabetes(p: PatientProfile) -> bool:
    # chỉ dùng số đói người dùng nhập; số ước lượng không được dùng để chẩn đoán
    return p.hba1c >= 6.5 or (p.fasting_glucose is not None and p.fasting_glucose >= 126)


def is_prediabetes(p: PatientProfile) -> bool:
    return 5.7 <= p.hba1c < 6.5 or (p.fasting_glucose is not None and 100 <= p.fasting_glucose < 126)


def infer_type(p: PatientProfile) -> str:
    """Xét nghiệm chưa làm (None) không cộng điểm."""
    onset = max(p.age - p.duration_years, 0)
    score = 2 * (p.autoantibody is True) + 2 * (p.c_peptide is not None and p.c_peptide < 0.6) + 1 * (onset < 30)
    return "T1" if score >= 3 else "T2"


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def complication_risks(p: PatientProfile) -> dict:
    d, h = p.duration_years, p.hba1c
    return {
        "retinopathy": _sigmoid(-8 + 0.6 * h + 0.1 * d + 0.01 * p.age),
        "nephropathy": _sigmoid(-7.5 + 0.5 * h + 0.08 * d + 0.02 * max(0, 60 - p.egfr) + 0.5 * p.hypertension),
        "neuropathy": _sigmoid(-8 + 0.5 * h + 0.1 * d + 0.02 * p.age),
        "label": "ước lượng theo quy tắc, không phải mô hình học máy",
    }


def classify_with_prob(p: PatientProfile, prob: float, thr: float) -> dict:
    inferred = False
    if p.diabetes_type_known:
        code, conf, source, risk = p.diabetes_type_known, 1.0, "đã chẩn đoán", "Đã chẩn đoán"
    elif meets_ada_diabetes(p):
        state = "DL vượt ngưỡng" if prob >= thr else "DL dưới ngưỡng"
        code, conf, source = infer_type(p), 1.0, f"tiêu chuẩn ADA ({state} - cần xét nghiệm xác nhận)"
        risk, inferred = "Đạt tiêu chuẩn chẩn đoán", True
    elif p.on_insulin:
        code, conf, source = infer_type(p), max(prob, 1 - prob), "đang dùng insulin, chưa nhập loại tiểu đường"
        risk, inferred = "Đang điều trị", True
    else:
        # ngưỡng sàng lọc có precision thấp: chỉ báo nguy cơ, không dán nhãn đái tháo đường
        screened = prob >= thr
        conf = max(prob, 1 - prob)
        if is_prediabetes(p):
            code, source, risk = "PRE", "DL + ngưỡng ADA", "Nguy cơ cao, cần xét nghiệm" if screened else "Tiền tiểu đường"
        else:
            code, source, risk = "NONE", "DL", "Nguy cơ cao, cần xét nghiệm" if screened else "Nguy cơ thấp"
    return {"code": code, "class_name": CLASS_NAMES.get(code, code), "confidence": round(float(conf), 3),
            "source": source, "risk_level": risk, "prob": round(float(prob), 3), "threshold": round(float(thr), 3),
            "type_low_confidence": inferred and p.c_peptide is None and p.autoantibody is None,
            "complications": complication_risks(p)}


def pad_window(readings, in_len: int = 12) -> list[float]:
    """Đệm đầu chuỗi bằng giá trị đầu tiên khi chưa đủ cửa sổ."""
    r = [float(x) for x in readings][-in_len:]
    return [r[0]] * (in_len - len(r)) + r


def conservative_forecast(window, lstm_vals, n_input: int, in_len: int = 12) -> dict:
    """Lấy mức thấp hơn giữa mô hình và ngoại suy: dự báo cao hơn thực tế nguy hiểm hơn thấp hơn."""
    w = np.asarray(window, dtype=float)
    tail = w[-4:]
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 2 else 0.0
    vals = np.clip(np.asarray(lstm_vals, dtype=float), 30, 450)
    horizon = len(vals)
    linear = w[-1] + slope * np.arange(1, horizon + 1)
    cons = np.clip(np.minimum(vals, linear), 30, 450)
    return {"values": [round(float(v), 1) for v in vals],
            "values_conservative": [round(float(v), 1) for v in cons],
            "minutes": [5 * (i + 1) for i in range(horizon)],
            "trend_per_5min": round(slope, 2), "n_input": n_input, "low_confidence": n_input < in_len}
