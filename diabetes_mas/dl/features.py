"""Mã hoá 13 đặc trưng. 'No Info' và gender 'Other' là toàn 0 (không coi là một giá trị đã biết)."""
import numpy as np
import pandas as pd

from ..schemas import PatientProfile

NUMERIC = ["age", "hypertension", "heart_disease", "bmi", "hba1c", "glucose"]
SMOKING = ["never", "former", "current", "ever", "not current"]
FEATURES = NUMERIC + ["gender_Female", "gender_Male"] + [f"smoking_{s.replace(' ', '_')}" for s in SMOKING]
N_NUMERIC = len(NUMERIC)


def encode_df(df: pd.DataFrame) -> np.ndarray:
    cols = [df["age"], df["hypertension"], df["heart_disease"], df["bmi"], df["HbA1c_level"],
            df["blood_glucose_level"], df["gender"] == "Female", df["gender"] == "Male"]
    cols += [df["smoking_history"] == s for s in SMOKING]
    return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols])


def encode_profile(p: PatientProfile, glucose: float) -> np.ndarray:
    sex = str(p.sex).upper()
    row = [p.age, float(p.hypertension), float(p.heart_disease), p.bmi, p.hba1c, glucose,
           float(sex.startswith("F")), float(sex.startswith("M"))]
    row += [float(p.smoking == s) for s in SMOKING]
    return np.array(row, dtype=np.float32)


def normalize(x: np.ndarray, mean, std) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    out[..., :N_NUMERIC] = (out[..., :N_NUMERIC] - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return out
