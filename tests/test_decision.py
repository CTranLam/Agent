from diabetes_mas.dl.decision import classify_with_prob, complication_risks, conservative_forecast, infer_type
from diabetes_mas.schemas import PatientProfile


def test_known_type_wins():
    assert classify_with_prob(PatientProfile(diabetes_type_known="T1"), .0, .09)["code"] == "T1"

def test_ada_net_catches_low_prob():
    r = classify_with_prob(PatientProfile(age=30, hba1c=6.7), .01, .09)
    assert r["code"] == "T2" and "ADA" in r["source"]

def test_prediabetes_and_none():
    assert classify_with_prob(PatientProfile(hba1c=5.8), .01, .09)["code"] == "PRE"
    assert classify_with_prob(PatientProfile(hba1c=5.1), .01, .09)["code"] == "NONE"

def test_missing_labs_score_zero():
    assert infer_type(PatientProfile(age=20)) == "T2"
    assert classify_with_prob(PatientProfile(age=20, hba1c=8), .9, .09)["type_low_confidence"] is True

def test_t1_inference():
    assert infer_type(PatientProfile(age=20, autoantibody=True, c_peptide=.1)) == "T1"

def test_complications_labelled_and_monotonic():
    a = complication_risks(PatientProfile(hba1c=6)); b = complication_risks(PatientProfile(hba1c=10))
    assert b["retinopathy"] > a["retinopathy"] and "quy tắc" in a["label"]

def test_conservative_takes_min_and_flags():
    f = conservative_forecast([100, 95, 90, 85], [200.0]*6, 4)
    assert f["values_conservative"][0] == 80 and f["trend_per_5min"] == -5
    assert f["low_confidence"] is True and f["minutes"][-1] == 30
