import numpy as np
from diabetes_mas import config as C
from diabetes_mas.policy.safety_rules import apply_safety, cap_total_bolus, check_meal_bolus
from diabetes_mas.schemas import PatientProfile


def test_apply_safety_invariants_random():
    rng = np.random.default_rng(42)
    for _ in range(5000):
        p = PatientProfile(on_insulin=bool(rng.random() < .8), isf=float(rng.uniform(10, 120)))
        g, fmin = float(rng.uniform(30, 500)), float(rng.uniform(30, 500))
        prop = float(rng.choice([0, .5, 1, 1.5, 2, 3, 4, 8]))
        mins = None if rng.random() < .4 else float(rng.uniform(0, 300))
        last = None if rng.random() < .4 else float(rng.uniform(0, 10))
        n = int(rng.integers(1, 30))
        age = None if rng.random() < .4 else float(rng.uniform(0, 60))
        d = apply_safety(p, g, fmin, prop, mins, "T1", n_readings=n, reading_age_min=age, last_insulin_units=last)
        ctx = (g, fmin, prop, mins, last, n, age, p.on_insulin, p.isf)
        assert 0 <= d.dose <= min(prop, C.MAX_CORRECTION_DOSE) + 1e-9, ctx
        if g < C.HYPO_L1 or fmin < C.HYPO_L1:
            assert d.dose == 0, ctx
        if not p.on_insulin:
            assert d.dose == 0, ctx
        if n < C.MIN_READINGS_FOR_DOSE:
            assert d.dose == 0, ctx
        if age is not None and age > C.READING_STALE_MIN:
            assert d.dose == 0, ctx
        if mins is not None and mins < C.INSULIN_ACTION_MIN and last is None and prop > 0:
            assert d.dose == 0, ctx
        if d.dose > 0:
            assert d.dose <= max(0.0, (g - C.TARGET_GLUCOSE) / p.isf) + 1e-9, ctx
            assert any(a.message.startswith("Liều chỉ mang tính THAM KHẢO") for a in d.alerts)


def test_meal_and_total_invariants_random():
    rng = np.random.default_rng(43)
    for _ in range(3000):
        p = PatientProfile(on_insulin=bool(rng.random() < .8), icr=float(rng.uniform(3, 30)))
        g, fmin = float(rng.uniform(30, 500)), float(rng.uniform(30, 500))
        carbs = None if rng.random() < .3 else float(rng.uniform(0, 300))
        age = None if rng.random() < .5 else float(rng.uniform(0, 60))
        meal, _ = check_meal_bolus(p, g, fmin, carbs, reading_age_min=age)
        ctx = (g, fmin, carbs, age, p.on_insulin, p.icr)
        assert 0 <= meal <= C.MAX_MEAL_BOLUS, ctx
        if carbs is None or not p.on_insulin or g < C.HYPO_L1 or fmin < C.HYPO_L1:
            assert meal == 0, ctx
        corr = float(rng.choice([0, 1, 2, 4]))
        c2, m2, _ = cap_total_bolus(corr, meal)
        assert c2 + m2 <= C.MAX_TOTAL_BOLUS and c2 <= corr and m2 <= meal, ctx
