import pytest
from diabetes_mas.policy.exercise_gate import exercise_gate
from diabetes_mas.policy.rl_insulin import QPolicy, evaluate, trend_formula, encode, reward_fn
from diabetes_mas.policy.safety_rules import apply_safety, cap_total_bolus, check_meal_bolus, hyper_alerts, iob_fraction
from diabetes_mas.schemas import AnalysisContext, PatientProfile

T1 = lambda **k: PatientProfile(on_insulin=True, isf=40, icr=12, **k)

def ctx_for(p, g, fmin=None, code="T1"):
    fmin = g if fmin is None else fmin
    return AnalysisContext(p, [g] * 12, {"code": code}, {"values_conservative": [fmin] * 6})


def test_iob_example_from_spec():
    assert iob_fraction(90) == 0.5
    d = apply_safety(T1(), 220, 200, 2.0, 90, "T1", n_readings=12, reading_age_min=0, last_insulin_units=3)
    assert d.dose == 0.5 and d.iob == 1.5

def test_iob_bounds():
    assert iob_fraction(0) == 1 and iob_fraction(180) == 0 and iob_fraction(500) == 0

@pytest.mark.parametrize("kw", [dict(g=50), dict(g=65), dict(fmin=60), dict(n=2), dict(n=None), dict(age=30), dict(mins=60, last=None)])
def test_rules_force_zero(kw):
    a = dict(g=220, fmin=200, n=12, age=0, mins=None, last=None) | kw
    d = apply_safety(T1(), a["g"], a["fmin"], 2.0, a["mins"], "T1", n_readings=a["n"],
                     reading_age_min=a["age"], last_insulin_units=a["last"])
    assert d.dose == 0 and d.blocked and d.rules_fired

def test_allowed_dose_has_reference_alert():
    d = apply_safety(T1(), 220, 200, 2.0, 240, "T1", n_readings=12)
    assert d.dose == 2.0 and any(a.message.startswith("Liều chỉ mang tính THAM KHẢO") for a in d.alerts)

def test_not_on_insulin_zero():
    assert apply_safety(PatientProfile(), 300, 280, 3.0, None, "T2", n_readings=12).dose == 0

def test_meal_requires_user_carbs():
    assert check_meal_bolus(T1(), 150, 140, None)[0] == 0
    assert check_meal_bolus(T1(), 150, 140, 60)[0] == 5.0
    assert check_meal_bolus(T1(), 60, 60, 60)[0] == 0

def test_meal_blocked_when_stale_or_zero():
    assert check_meal_bolus(T1(), 150, 140, 60, reading_age_min=40)[0] == 0
    assert check_meal_bolus(T1(), 150, 140, 0)[0] == 0

def test_meal_capped():
    assert check_meal_bolus(T1(), 150, 140, 300)[0] == 10.0

def test_total_cap_cuts_correction_first():
    assert cap_total_bolus(4.0, 10.0)[:2] == (2.0, 10.0)

def test_hyper_alerts_without_insulin():
    assert any(a.level == "critical" for a in hyper_alerts(420, False, "T2"))
    assert any("ceton" in a.message for a in hyper_alerts(260, True, "T1"))
    assert not hyper_alerts(260, False, "T2")

def test_exercise_gate_t1_265():
    assert not exercise_gate(ctx_for(T1(), 265)).allowed

def test_exercise_gate_levels():
    assert not exercise_gate(ctx_for(PatientProfile(), 120, fmin=60)).allowed
    assert not exercise_gate(ctx_for(PatientProfile(), 310, code="T2")).allowed
    assert exercise_gate(ctx_for(PatientProfile(), 260, code="T2")).allowed
    assert exercise_gate(ctx_for(PatientProfile(), 85, fmin=85)).intensity == "nhẹ"
    assert exercise_gate(ctx_for(PatientProfile(age=80), 120)).intensity == "nhẹ"
    assert exercise_gate(ctx_for(PatientProfile(), 120)).intensity == "trung bình"

def test_state_space_size():
    assert encode(10, -99, 1) == (0, 0, 0) and encode(999, 99, 999) == (16, 6, 8)

def test_reward_penalizes_hypo_more():
    assert reward_fn(110) == 0 and reward_fn(50) < reward_fn(65) < reward_fn(200)

def test_unseen_state_suggests_zero():
    q = QPolicy(); q.train(50_000, seed=0)
    assert q.suggest(200, 0, 5) == 0  # ISF ngoài vùng đã huấn luyện -> không đoán

def test_rl_close_to_trend_formula():
    q = QPolicy(); q.train(300_000, seed=0)
    rl, none, formula = evaluate(q.suggest), evaluate(lambda *a: 0.0), evaluate(trend_formula)
    assert rl["tir"] > none["tir"] + 0.2 and rl["tir"] > formula["tir"] - 0.07
