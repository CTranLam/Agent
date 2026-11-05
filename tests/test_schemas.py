import json
from diabetes_mas.schemas import PatientProfile, AnalysisContext, Report

def test_bmi():
    assert round(PatientProfile(weight_kg=70, height_cm=165).bmi, 1) == 25.7

def test_defaults_not_shared():
    a, b = PatientProfile(), PatientProfile()
    a.medications.append("x")
    assert b.medications == []

def test_unknown_labs_stay_none():
    p = PatientProfile()
    assert p.c_peptide is None and p.autoantibody is None and p.fasting_glucose is None

def test_context_shortcuts():
    ctx = AnalysisContext(PatientProfile(), [100.0, 90.0], {"code": "T2"},
                          {"values_conservative": [80.0, 75.0]})
    assert (ctx.glucose, ctx.n_readings, ctx.forecast_min, ctx.diabetes_type) == (90.0, 2, 75.0, "T2")

def test_report_to_dict_is_json_safe():
    json.dumps(Report("thường quy", [], {}, {}, {}, [], [], None, None, [], "d").to_dict())
