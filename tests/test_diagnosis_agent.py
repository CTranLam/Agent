import pytest
from diabetes_mas.agents.diagnosis_agent import DiagnosisAgent
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.dl.decision import classify_with_prob
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.scenarios import SCENARIOS
from diabetes_mas.schemas import AnalysisContext, PatientProfile
from tests.fakes import FakePredictor

kb = load_kb(); rag = RAG(kb)
THR = 0.09


def run(p, prob):
    c = classify_with_prob(p, prob, THR)
    return DiagnosisAgent(kb, rag).run(AnalysisContext(p, [100.0], c, FakePredictor().forecast([100.0])))


def test_reports_model_output_and_conclusion():
    r = run(PatientProfile(hba1c=5.1), 0.02)
    d = r.details
    assert (d["code"], d["prob"], d["threshold"], d["screened"]) == ("NONE", 0.02, THR, False)
    assert "2,0%" in r.summary and "9,0%" in r.summary and "Không tiểu đường" in r.summary
    assert r.evidence and "bác sĩ" in r.summary

def test_lists_criteria_actually_met():
    d = run(PatientProfile(hba1c=7.2, fasting_glucose=150), 0.5).details
    assert d["meets_ada"] and len(d["criteria"]) == 2 and "7,2" in d["criteria"][0]

def test_estimated_or_missing_fasting_is_not_a_criterion():
    d = run(PatientProfile(hba1c=6.7), 0.5).details
    assert len(d["criteria"]) == 1 and "đường huyết đói" in " ".join(d["missing_tests"]).lower()

def test_ada_met_but_model_below_threshold_warns():
    r = run(PatientProfile(age=30, hba1c=6.7), 0.01)
    assert r.details["code"] == "T2" and any(a.level == "warning" and "dưới ngưỡng" in a.message for a in r.alerts)

def test_screen_positive_without_criteria_is_not_called_diabetes():
    r = run(PatientProfile(hba1c=5.1), 0.5)
    assert r.details["code"] == "NONE" and any("xét nghiệm" in x for x in r.actions)
    assert any("vượt ngưỡng" in a.message for a in r.alerts)

def test_prediabetes_asks_to_repeat_test():
    r = run(PatientProfile(hba1c=5.9), 0.01)
    assert r.details["code"] == "PRE" and any("nhắc lại" in x for x in r.actions)

def test_inferred_type_lists_missing_labs():
    d = run(PatientProfile(age=40, hba1c=8), 0.9).details
    assert d["inferred_type"] and {"C-peptide", "kháng thể tự miễn"} <= set(d["missing_tests"])

def test_known_type_is_kept_and_model_cannot_override():
    r = run(PatientProfile(diabetes_type_known="T1", hba1c=5.0), 0.01)
    assert r.details["code"] == "T1" and not r.details["inferred_type"] and not r.alerts
    assert any("giữ nguyên" in x for x in r.actions)

def test_insulin_without_type_asks_to_enter_type():
    r = run(PatientProfile(on_insulin=True, hba1c=5.4), 0.01)
    assert any("loại tiểu đường" in x for x in r.actions)

@pytest.mark.parametrize("key", SCENARIOS)
def test_runs_first_in_every_plan_with_no_audit_findings(key):
    class Fixed:
        def suggest(self, g, t, i): return 2.0
    s = SCENARIOS[key]
    rep = Orchestrator(FakePredictor(s.get("prob", 0.01)), kb, rag, Fixed()).analyze(
        s["profile"], s["readings"], planned_carbs=s.get("planned_carbs"),
        minutes_since_insulin=s.get("minutes_since_insulin"), last_insulin_units=s.get("last_insulin_units"))
    assert rep.steps[0] == "diagnosis" and rep.results["diagnosis"].details["code"] == rep.classification["code"]
    assert rep.audit_findings == []
