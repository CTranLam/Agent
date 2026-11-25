"""Mỗi test tái hiện một phát hiện của lần review toàn nhánh."""
import pytest
from fastapi.testclient import TestClient
from diabetes_mas.agents.auditor import audit
from diabetes_mas.agents.glucose_agent import GlucoseAgent
from diabetes_mas.agents.medication_agent import MedicationAgent
from diabetes_mas.agents.nutrition_agent import NutritionAgent
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.api import create_app
from diabetes_mas.dl.decision import classify_with_prob
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.policy.safety_rules import apply_safety, check_meal_bolus
from diabetes_mas.scenarios import SCENARIOS
from diabetes_mas.schemas import AgentResult, AnalysisContext, PatientProfile
from tests.fakes import FakePredictor

kb = load_kb(); rag = RAG(kb)
T1 = lambda **k: PatientProfile(on_insulin=True, isf=40, icr=12, **k)
flat = lambda g, n=12: [float(g)] * n


class Fixed:
    def __init__(self, dose=2.0): self.dose = dose
    def suggest(self, g, t, i): return self.dose

def med(): return MedicationAgent(kb, rag, Fixed())
def orch(): return Orchestrator(FakePredictor(), kb, rag, Fixed())
def client(): return TestClient(create_app(orch(), rag))
def ctx(p=None, readings=None, code="T2", carbs=None, **kw):
    r = readings or flat(120)
    return AnalysisContext(p or PatientProfile(), r, {"code": code, "complications": {}}, FakePredictor().forecast(r), planned_carbs=carbs, **kw)
def post(**kw): return client().post("/analyze", json={"profile": {}} | kw)


# C1: dấu phẩy thập phân
def test_decimal_comma_mmol_is_one_value():
    j = post(series_text="2,9", unit="mmol/L").json()
    assert j["results"]["glucose"]["details"]["current"] == pytest.approx(52.2, abs=.1) and j["plan"].startswith("KHẨN")

def test_decimal_comma_mmol_series():
    assert post(series_text="2,9, 3,1", unit="mmol/L").json()["readings"] == pytest.approx([52.25, 55.85], abs=.01)

def test_mgdl_commas_are_separators():
    assert post(series_text="120,125, 130").json()["readings"] == [120, 125, 130]

def test_mgdl_decimal_comma_rejected_as_unit():
    r = post(series_text="5,6"); assert r.status_code == 422 and "mmol/L" in r.json()["detail"]

@pytest.mark.parametrize("kw", [{"series_text": "abc"}, {}, {"series_text": " , "}, {"series_text": ", ".join(["120"] * 300)}])
def test_bad_series_is_422(kw):
    assert post(**kw).status_code == 422


# C2: không rõ lần tiêm thì không có liều bù
@pytest.mark.parametrize("last", [None, 8])
def test_unknown_last_injection_blocks(last):
    d = apply_safety(T1(), 222, 210, 2.5, None, "T1", n_readings=12, last_insulin_units=last)
    assert d.dose == 0 and d.blocked and any("lần tiêm" in a.message for a in d.alerts)

def test_known_no_recent_injection_allows_dose():
    assert apply_safety(T1(), 222, 210, 2.5, 180, "T1", n_readings=12).dose > 0

def test_auditor_catches_dose_without_injection_time():
    results = {"medication": AgentResult("m", "", {"correction": 2.0, "meal": 0.0})}
    assert "A3" in {f.rule for f in audit(ctx(T1(), readings=flat(200)), results)}


# I1: bị chặn không hiện như "0 đv"
def test_blocked_dose_reports_reason_not_zero():
    r = med().run(ctx(T1(), readings=[300.0], code="T1", minutes_since_insulin=240))
    assert r.details["blocked_reason"] and r.details["proposed"] is None and r.details["correction"] == 0
    assert "bù 0" not in r.summary and "Không đề xuất liều bù" in r.summary


# I2: ISF/ICR phải được nhập, trẻ em không có liều
def test_missing_isf_blocks_and_missing_icr_no_meal():
    p = PatientProfile(on_insulin=True)
    assert apply_safety(p, 220, 200, 2.0, 240, "T1", n_readings=12).dose == 0
    dose, notes = check_meal_bolus(p, 150, 140, 60)
    assert dose == 0 and any("ICR" in n for n in notes)

def test_missing_isf_end_to_end_no_dose():
    d = med().run(ctx(PatientProfile(on_insulin=True), readings=flat(220), code="T1", carbs=60, minutes_since_insulin=240)).details
    assert d["correction"] == 0 and d["meal"] == 0 and d["blocked_reason"]

def test_child_gets_no_dose():
    p = T1(age=8)
    assert apply_safety(p, 320, 300, 3.0, 240, "T1", n_readings=12).dose == 0
    assert check_meal_bolus(p, 320, 300, 120)[0] == 0


# I3: thấp thật không bị coi là lỗi đơn vị
def test_very_low_message_also_gives_emergency_advice():
    r = post(readings=[30]); assert r.status_code == 422 and "cấp cứu" in r.json()["detail"]

def test_mmol_unit_confirmed_low_is_emergency():
    r = post(readings=[1.8], unit="mmol/L")
    assert r.status_code == 200 and r.json()["plan"].startswith("KHẨN")


# I4: nhiều thuốc trong một ô
@pytest.mark.parametrize("txt", ["Metformin 500mg; Pioglitazone 15mg", "Metformin 850mg và Pioglitazone 15mg", "Metformin 500mg + Pioglitazone 15mg"])
def test_all_drugs_in_one_entry_reviewed(txt):
    r = med().run(ctx(PatientProfile(medications=[txt], heart_failure=True)))
    assert set(r.details["drugs"]) == {"metformin", "pioglitazone"} and any(a.level == "critical" for a in r.alerts)

def test_leftover_text_reported():
    assert med().run(ctx(PatientProfile(medications=["Metformin 850mg và thuoc la"]))).details["unknown_drugs"] == ["thuoc la"]


# I5: ngoài phạm vi và câu hạ đường huyết
@pytest.mark.parametrize("q", ["đường cao tốc Hà Nội", "chứng khoán thế nào", "tiền lương", "đi chung xe", "giá đường hôm nay"])
def test_more_off_topic_returns_nothing(q):
    assert rag.retrieve(q) == []

@pytest.mark.parametrize("q", ["tụt đường phải làm sao", "đường huyết thấp phải làm sao", "hạ đường huyết"])
def test_hypo_questions_surface_treatment(q):
    assert any("15-15" in e.title for e in rag.retrieve(q))


# I7: ngưỡng sàng lọc không phải chẩn đoán
def test_screening_positive_below_ada_is_not_diabetes():
    a = classify_with_prob(PatientProfile(hba1c=5.8), .14, .09)
    assert a["code"] == "PRE" and "Nguy cơ cao" in a["risk_level"]
    b = classify_with_prob(PatientProfile(hba1c=5.1), .5, .09)
    assert b["code"] == "NONE" and "xét nghiệm" in b["risk_level"]

def test_ada_source_when_prob_above_threshold():
    a = classify_with_prob(PatientProfile(age=30, hba1c=6.7), .5, .09)
    assert a["code"] == "T2" and "ADA" in a["source"]


# I8: đang điều trị
def test_insulin_user_is_treated_as_diabetic():
    a = classify_with_prob(PatientProfile(on_insulin=True, hba1c=5.4, age=50), .01, .09)
    assert a["code"] in ("T1", "T2") and a["type_low_confidence"] is True

def test_diabetes_drug_with_normal_labs_is_flagged():
    r = med().run(ctx(PatientProfile(medications=["metformin"], hba1c=5.6), code="NONE"))
    assert any("loại tiểu đường" in a.message for a in r.alerts)


# I9: dị ứng
@pytest.mark.parametrize("allergy,banned", [("hải sản", "Tôm hấp"), ("seafood", "Cá hấp"), ("trứng gà", "Trứng luộc"), ("đậu", "Đậu phụ sốt cà")])
def test_allergy_groups_filter(allergy, banned):
    agent = NutritionAgent(kb, rag)
    assert banned not in {f["name"] for f in agent._allowed(PatientProfile(allergies=[allergy]))}

def test_unmatched_allergy_warns():
    r = NutritionAgent(kb, rag).run(ctx(PatientProfile(allergies=["xyz"]), code="NONE"))
    assert any("xyz" in a.message for a in r.alerts)


# I11: thuốc
def test_pregnancy_flags_oral_drugs():
    r = med().run(ctx(PatientProfile(pregnant=True, medications=["gliclazide", "empagliflozin", "sitagliptin", "metformin"])))
    msg = " ".join(a.message.lower() for a in r.alerts)
    assert all(d in msg for d in ("gliclazide", "empagliflozin", "sitagliptin", "metformin"))

def test_gliclazide_contraindicated_severe_renal():
    r = med().run(ctx(PatientProfile(egfr=25, medications=["gliclazide"])))
    assert any(a.level == "critical" and "gliclazide" in a.message.lower() for a in r.alerts)

def test_substitutes_respect_type_and_pregnancy():
    r = med().run(ctx(PatientProfile(pregnant=True, heart_failure=True, medications=["pioglitazone"])))
    assert not any("empagliflozin" in a or "sitagliptin" in a for a in r.actions)
    r = med().run(ctx(PatientProfile(heart_failure=True, medications=["pioglitazone"]), code="T1"))
    assert not any("sitagliptin" in a or "empagliflozin" in a for a in r.actions)


# I12: Auditor
def tampered(correction=3.0, meal=0.0):
    return {"medication": AgentResult("m", f"Liều tham khảo: bù {correction:g} đv, bữa ăn {meal:g} đv.", {"correction": correction, "meal": meal},
                                      [f"Liều bù tham khảo: {correction:g} đv.", f"Liều bữa ăn tham khảo: {meal:g} đv."])}

def test_audit_rewrites_text_after_cancel():
    r = tampered(); audit(ctx(T1(), readings=flat(60)), r)
    m = r["medication"]
    assert not any(a.startswith("Liều bù tham khảo") for a in m.actions) and "Auditor" in m.summary

def test_audit_rewrites_text_after_cap():
    r = tampered(correction=8.0); audit(ctx(T1(), readings=flat(200), minutes_since_insulin=240), r)
    assert r["medication"].details["correction"] == 4 and not any("8 đv" in a for a in r["medication"].actions)

def test_audit_meal_cap():
    r = tampered(correction=0.0, meal=11.0); f = audit(ctx(T1(), readings=flat(200), carbs=200, minutes_since_insulin=240), r)
    assert r["medication"].details["meal"] == 10 and "A4" in {x.rule for x in f}

def test_audit_meal_stale():
    r = tampered(correction=0.0, meal=5.0); f = audit(ctx(T1(), readings=flat(200), carbs=60, reading_age_min=40, minutes_since_insulin=240), r)
    assert r["medication"].details["meal"] == 0 and "A3" in {x.rule for x in f}

def test_audit_dose_for_non_insulin_patient():
    r = tampered(); f = audit(ctx(PatientProfile(), readings=flat(200)), r)
    assert r["medication"].details["correction"] == 0 and "A8" in {x.rule for x in f}


# M1 (xếp lại thành Important): rất cao mà đang giảm nhanh
def test_high_and_falling_keeps_hyper_alerts():
    c = ctx(readings=flat(390)); c.forecast = {**c.forecast, "values_conservative": [60.0] * 6}
    r = GlucoseAgent(kb, rag).run(c)
    assert "trên mục tiêu" in r.summary and any("300" in a.message for a in r.alerts) and any("Dự báo" in a.message for a in r.alerts)


def test_scenario_8_has_injection_time_so_dose_is_possible():
    assert SCENARIOS["8"]["minutes_since_insulin"] >= 180
