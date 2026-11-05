import numpy as np, pandas as pd, torch
from diabetes_mas.dl.features import FEATURES, encode_df, encode_profile
from diabetes_mas.dl.models import DiabetesNet
from diabetes_mas.dl.train import split_dedup, pick_threshold
from diabetes_mas.schemas import PatientProfile


def fake_df(n=400):
    rng = np.random.default_rng(0)
    h = rng.uniform(4, 9, n).round(1)
    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female", "Other"], n),
        "age": rng.integers(1, 80, n).astype(float), "hypertension": rng.integers(0, 2, n),
        "heart_disease": rng.integers(0, 2, n),
        "smoking_history": rng.choice(["never", "former", "current", "ever", "not current", "No Info"], n),
        "bmi": rng.uniform(15, 40, n).round(2), "HbA1c_level": h,
        "blood_glucose_level": rng.integers(80, 300, n), "diabetes": (h >= 6.5).astype(int)})


def test_feature_count_and_alignment():
    assert encode_df(fake_df()).shape[1] == 13 == len(FEATURES)
    row = pd.DataFrame([{"gender": "Male", "age": 30, "hypertension": 0, "heart_disease": 0,
                         "smoking_history": "never", "bmi": 25.0, "HbA1c_level": 6.0,
                         "blood_glucose_level": 110, "diabetes": 0}])
    p = PatientProfile(age=30, sex="M", weight_kg=25.0, height_cm=100, hba1c=6.0, smoking="never")
    assert np.allclose(encode_df(row)[0], encode_profile(p, 110.0))


def test_no_info_is_all_zero_smoking():
    i = [k for k, f in enumerate(FEATURES) if f.startswith("smoking_")]
    assert encode_df(fake_df().assign(smoking_history="No Info"))[:, i].sum() == 0


def test_output_is_logit_shape():
    assert DiabetesNet(13)(torch.zeros(5, 13)).shape == (5,)


def test_split_has_no_row_overlap():
    tr, va, te = split_dedup(pd.concat([fake_df(), fake_df()]))
    assert not set(map(tuple, tr.values)) & set(map(tuple, te.values))
    assert len(tr) + len(va) + len(te) == 400


def test_threshold_reaches_recall():
    y, p = np.array([1, 1, 1, 0, 0]), np.array([.9, .8, .2, .5, .1])
    t = pick_threshold(y, p, 0.9)
    assert ((p >= t)[y == 1]).mean() >= 0.9 and t == 0.2
