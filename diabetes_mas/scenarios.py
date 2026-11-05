"""Các ca mẫu để demo; cũng là test tích hợp."""
import numpy as np

from .schemas import PatientProfile


def cgm(start: float, end: float, n: int = 12, seed: int = 0) -> list[float]:
    noise = np.random.default_rng(seed).normal(0, 1.0, n)
    return [round(float(x), 1) for x in np.linspace(start, end, n) + noise]


SCENARIOS = {
    "1": {"title": "Khoẻ mạnh, chỉ có 1 số đo",
          "profile": PatientProfile(name="Ca 1", age=55, sex="M", weight_kg=60, height_cm=170, hba1c=5.1),
          "readings": [100.0]},
    "2": {"title": "Tiền tiểu đường",
          "profile": PatientProfile(name="Ca 2", age=50, sex="M", weight_kg=51, height_cm=170, hba1c=5.8),
          "readings": [124.0]},
    "3": {"title": "HbA1c 6,7%: đạt tiêu chuẩn ADA", "prob": 0.01,
          "profile": PatientProfile(name="Ca 3", age=30, sex="M", weight_kg=55, height_cm=163, hba1c=6.7),
          "readings": [113.0]},
    "4": {"title": "Type 1 dùng insulin, sắp ăn, vừa tiêm 90 phút trước",
          "profile": PatientProfile(name="Ca 4", age=27, sex="M", weight_kg=66, height_cm=175, hba1c=7.6,
                                    fasting_glucose=130, duration_years=9, c_peptide=0.1, autoantibody=True, egfr=110,
                                    on_insulin=True, medications=["Insulin glargine", "Insulin aspart"], isf=40, icr=12,
                                    diabetes_type_known="T1"),
          "readings": cgm(205, 215, seed=9), "planned_carbs": 60, "minutes_since_insulin": 90,
          "last_insulin_units": 3},
    "5": {"title": "Cao tuổi dùng gliclazide, suy thận, hạ đường huyết nặng",
          "profile": PatientProfile(name="Ca 5", age=80, sex="F", weight_kg=45, height_cm=150, hba1c=7.0, egfr=25,
                                    duration_years=15, medications=["Gliclazide 60mg"], diabetes_type_known="T2"),
          "readings": cgm(70, 48, seed=5)},
    "6": {"title": "Dùng metformin, đường huyết rất cao",
          "profile": PatientProfile(name="Ca 6", age=58, sex="M", weight_kg=78, height_cm=168, hba1c=11.2,
                                    medications=["Metformin 850mg"], diabetes_type_known="T2"),
          "readings": cgm(390, 420, seed=6)},
    "7": {"title": "Suy tim đang dùng pioglitazone",
          "profile": PatientProfile(name="Ca 7", age=66, sex="F", weight_kg=70, height_cm=155, hba1c=7.9,
                                    heart_failure=True, heart_disease=True, medications=["Pioglitazone 15mg", "Metformin 500mg"],
                                    diabetes_type_known="T2"),
          "readings": cgm(150, 160, seed=7)},
    "8": {"title": "Type 1, đường huyết 265: chưa được tập",
          "profile": PatientProfile(name="Ca 8", age=24, sex="F", weight_kg=55, height_cm=160, hba1c=8.8, on_insulin=True,
                                    medications=["Insulin aspart", "Insulin glargine"], isf=45, icr=12, diabetes_type_known="T1"),
          "readings": cgm(250, 265, seed=8), "minutes_since_insulin": 240},
}
