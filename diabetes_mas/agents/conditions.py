"""Điều kiện lâm sàng trong KB là dữ liệu; bảng này đánh giá chúng trên hồ sơ bệnh nhân."""
from ..schemas import PatientProfile

EVALUATORS = {
    "egfr_lt_20": lambda p, t: p.egfr < 20,
    "egfr_lt_30": lambda p, t: p.egfr < 30,
    "egfr_lt_45": lambda p, t: p.egfr < 45,
    "egfr_lt_60": lambda p, t: p.egfr < 60,
    "heart_failure": lambda p, t: bool(p.heart_failure),
    "pregnancy": lambda p, t: bool(p.pregnant),
    "mtc_men2": lambda p, t: bool(p.mtc_men2_history),
    "pancreatitis_history": lambda p, t: bool(p.pancreatitis_history),
    "age_ge_75": lambda p, t: p.age >= 75,
    "type1_dka": lambda p, t: t == "T1",
}


def evaluate(code: str, profile: PatientProfile, dtype: str) -> bool | None:
    """True/False nếu đánh giá được; None nếu mã chưa có bộ đánh giá (không được coi là 'không xảy ra')."""
    fn = EVALUATORS.get(code)
    return None if fn is None else bool(fn(profile, dtype))
