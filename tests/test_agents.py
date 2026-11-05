from diabetes_mas.agents.conditions import evaluate
from diabetes_mas.agents.exercise_agent import ExerciseAgent
from diabetes_mas.agents.glucose_agent import GlucoseAgent
from diabetes_mas.agents.medication_agent import MedicationAgent
from diabetes_mas.agents.nutrition_agent import NutritionAgent, carb_per_meal, energy_kcal
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.schemas import AnalysisContext, PatientProfile
from tests.fakes import FakePredictor

kb = load_kb(); rag = RAG(kb)
T1 = lambda **k: PatientProfile(on_insulin=True, isf=40, icr=12, **k)
flat = lambda g, n=12: [float(g)] * n


class Fixed:
    def __init__(self, dose=2.0): self.dose = dose
    def suggest(self, g, t, i): return self.dose

def med_agent(): return MedicationAgent(kb, rag, Fixed())

def ctx(p=None, readings=None, code="T2", carbs=None, **kw):
    r = readings or flat(120)
    return AnalysisContext(p or PatientProfile(), r, {"code": code, "complications": {}},
                           FakePredictor().forecast(r), planned_carbs=carbs, **kw)


def test_kcal_examples_from_spec():
    p = PatientProfile(age=50, sex="M", weight_kg=70, height_cm=165)
    assert energy_kcal(p, "T2")[0] == 1500 and carb_per_meal(p, "T2")[0] == 60  # sàn 1500 kcal cho nam
    big = PatientProfile(age=50, sex="M", weight_kg=90, height_cm=170)
    assert energy_kcal(big, "T2")[0] == 1730
    assert energy_kcal(PatientProfile(age=60, sex="F", weight_kg=62, height_cm=150), "T2")[0] == 1200
    q = PatientProfile(age=55, sex="M", weight_kg=60, height_cm=170)
    assert energy_kcal(q, "NONE")[0] == 1810 and carb_per_meal(q, "NONE")[0] == 60

def test_no_weight_loss_for_elderly_or_t1():
    p = PatientProfile(age=80, sex="M", weight_kg=80, height_cm=165)
    assert energy_kcal(p, "T2")[1] == "duy trì cân nặng"
    assert energy_kcal(PatientProfile(weight_kg=80), "T1")[1] == "duy trì cân nặng"

def test_no_tir_with_one_reading():  # Review Focus 2
    d = GlucoseAgent(kb, rag).run(ctx(readings=[100.])).details
    assert d["tir_pct"] is None and d["tbr_pct"] is None and d["cv_pct"] is None

def test_tir_with_enough_readings():
    d = GlucoseAgent(kb, rag).run(ctx(readings=[100.] * 6 + [200.] * 6)).details
    assert d["tir_pct"] == 50 and d["tar_pct"] == 50 and d["tbr_pct"] == 0

def test_hypo_gives_fast_carb_foods_and_critical():
    r = GlucoseAgent(kb, rag).run(ctx(readings=flat(48)))
    assert any(a.level == "critical" for a in r.alerts)
    assert any("15" in a and "Viên glucose" in a for a in r.actions)

def test_stale_and_sensor_jump_flagged():
    assert any("cũ" in a.message for a in GlucoseAgent(kb, rag).run(ctx(reading_age_min=45)).alerts)
    assert any("cảm biến" in a.message for a in GlucoseAgent(kb, rag).run(ctx(readings=[100.] * 11 + [180.])).alerts)

def test_unknown_condition_is_none_not_false():
    assert evaluate("khong_co", PatientProfile(), "T2") is None

def test_every_kb_condition_has_evaluator():
    for d in load_kb().drugs:
        for c in d["contraindications"] + d["cautions"]:
            assert evaluate(c, PatientProfile(), "T2") is not None, c

def test_unknown_drug_reported():  # Review Focus 3
    r = med_agent().run(ctx(PatientProfile(medications=["Metformin 850mg", "thuoc la"])))
    assert r.details["unknown_drugs"] == ["thuoc la"]
    assert any("dược sĩ" in a.message for a in r.alerts)

def test_contraindication_and_alternative():
    r = med_agent().run(ctx(PatientProfile(medications=["pioglitazone"], heart_failure=True), code="T2"))
    assert any(a.level == "critical" for a in r.alerts)
    assert any("sitagliptin" in s.lower() or "linagliptin" in s.lower() for s in r.actions)

def test_alternative_not_suggested_if_already_taken():
    r = med_agent().run(ctx(PatientProfile(medications=["pioglitazone", "sitagliptin"], heart_failure=True)))
    assert not any("thay pioglitazone bằng sitagliptin" in s.lower() for s in r.actions)

def test_interaction_alert():
    r = med_agent().run(ctx(T1(medications=["gliclazide", "insulin aspart"]), code="T2"))
    assert any("Gliclazide cùng insulin" in a.message for a in r.alerts)

def test_no_meal_dose_without_user_carbs():
    assert med_agent().run(ctx(T1(), readings=flat(200), code="T1", carbs=None)).details["meal"] == 0

def test_doses_go_through_rules():
    d = med_agent().run(ctx(T1(), readings=flat(205), code="T1", carbs=60, minutes_since_insulin=90,
                            last_insulin_units=3)).details
    assert (d["proposed"], d["correction"], d["meal"]) == (2.0, 0.5, 5.0)

def test_review_only_has_no_dose():
    c = ctx(T1(), readings=flat(48), code="T1", carbs=60); c.shared["review_only"] = True
    d = med_agent().run(c).details
    assert d["correction"] is None and d["meal"] is None

def test_not_on_insulin_no_dose():
    d = med_agent().run(ctx(PatientProfile(medications=["metformin"]), readings=flat(250))).details
    assert d["correction"] == 0 and d["meal"] == 0

def test_menu_respects_filters_and_is_deterministic():
    p = PatientProfile(vegetarian=True, allergies=["tôm"], hypertension=True)
    a, b = (NutritionAgent(kb, rag).run(ctx(p, code="NONE")) for _ in range(2))
    assert a.details["menu"] == b.details["menu"]
    names = {i["name"] for i in a.details["menu"]["items"]}
    by = {f["name"]: f for f in kb.foods}
    assert names and all(by[n]["vegetarian"] and not by[n]["high_sodium"] and "tôm" not in by[n]["allergens"] for n in names)

def test_menu_within_budget():
    r = NutritionAgent(kb, rag).run(ctx(PatientProfile(age=55, weight_kg=60, height_cm=170), code="NONE"))
    assert r.details["menu"]["total_carb"] <= 60

def test_no_rice_at_breakfast():
    for seed_age in range(20, 80, 5):
        r = NutritionAgent(kb, rag).run(ctx(PatientProfile(age=seed_age), code="NONE", meal_type="breakfast"))
        assert not any("Cơm" in i["name"] for i in r.details["menu"]["items"])

def test_no_walk_advice_when_gate_closed():
    r = NutritionAgent(kb, rag).run(ctx(T1(), readings=flat(265), code="T1"))
    assert not any("đi bộ" in a.lower() for a in r.actions)

def test_exercise_agent_gate_closed_no_plan():
    r = ExerciseAgent(kb, rag).run(ctx(T1(), readings=flat(265), code="T1"))
    assert r.details["allowed"] is False and not any("150 phút" in a for a in r.actions)

def test_exercise_agent_cardiac_warning():
    r = ExerciseAgent(kb, rag).run(ctx(PatientProfile(heart_disease=True)))
    assert any("đau ngực" in a for a in r.actions)
