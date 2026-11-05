import numpy as np, pytest
from diabetes_mas.agents.auditor import audit
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.scenarios import SCENARIOS
from diabetes_mas.schemas import AgentResult, AnalysisContext, PatientProfile
from tests.fakes import FakePredictor

kb = load_kb(); rag = RAG(kb)
T1 = lambda **k: PatientProfile(on_insulin=True, isf=40, icr=12, **k)
flat = lambda g, n=12: [float(g)] * n


class Fixed:
    def __init__(self, dose=2.0): self.dose = dose
    def suggest(self, g, t, i): return self.dose

def orch(prob=0.01): return Orchestrator(FakePredictor(prob), kb, rag, Fixed())

def run(key):
    s = SCENARIOS[key]
    return orch(s.get("prob", 0.01)).analyze(
        s["profile"], s["readings"], planned_carbs=s.get("planned_carbs"),
        minutes_since_insulin=s.get("minutes_since_insulin"), last_insulin_units=s.get("last_insulin_units"))

def ctx(p=None, readings=None, code="T2", **kw):
    r = readings or flat(120)
    return AnalysisContext(p or PatientProfile(), r, {"code": code, "complications": {}}, FakePredictor().forecast(r), **kw)


def test_mmol_rejected():  # Review Focus 1
    with pytest.raises(ValueError, match="mmol/L"):
        orch().analyze(PatientProfile(), [5.6, 5.8, 6.0])

def test_out_of_range_rejected():
    with pytest.raises(ValueError): orch().analyze(PatientProfile(), [100, 700])
    with pytest.raises(ValueError): orch().analyze(PatientProfile(), [])
    with pytest.raises(ValueError): orch().analyze(PatientProfile(), [100, float("nan")])

def test_scenario_1():
    r = run("1")
    assert r.classification["code"] == "NONE" and r.results["glucose"].details["tir_pct"] is None
    assert r.results["nutrition"].details["budget_g"] == 60 and r.results["nutrition"].details["energy_kcal"] == 1810
    assert any("Chất lượng dữ liệu" in a.message for a in r.alerts)

def test_scenario_2_and_3_classification():
    assert run("2").classification["code"] == "PRE"
    r = run("3"); assert r.classification["code"] == "T2" and "ADA" in r.classification["source"]

def test_scenario_4_doses():
    r = run("4")   # policy giả luôn đề xuất 2.0
    assert (r.results["medication"].details["correction"], r.results["medication"].details["meal"]) == (0.5, 5.0)

def test_scenario_5_emergency_keeps_drug_review():
    r = run("5")
    assert r.plan.startswith("KHẨN CẤP") and "nutrition" not in r.results and "exercise" not in r.results
    assert r.results["medication"].details["correction"] is None
    assert any("gliclazide" in a.message.lower() for a in r.alerts)

def test_scenario_6_hyper_emergency():
    r = run("6")
    assert r.plan.startswith("KHẨN CẤP") and r.alerts[0].level == "critical"

def test_scenario_7_substitute():
    r = run("7")
    assert any(a.level == "critical" and "pioglitazone" in a.message.lower() for a in r.alerts)
    assert any("thay pioglitazone" in s for s in r.results["medication"].actions)

def test_scenario_8_no_walk_anywhere():
    r = run("8")
    assert r.results["exercise"].details["allowed"] is False
    assert not any("đi bộ" in a.lower() for res in r.results.values() for a in res.actions)

@pytest.mark.parametrize("key", SCENARIOS)
def test_auditor_silent_on_demo(key):
    assert run(key).audit_findings == []

def test_alerts_sorted_and_deduped():
    levels = [a.level for a in run("5").alerts]
    order = {"critical": 0, "warning": 1, "info": 2}
    assert levels == sorted(levels, key=order.get)
    msgs = [a.message for a in run("5").alerts]
    assert len(msgs) == len(set(msgs))

def test_auditor_catches_forced_violation():
    results = {"medication": AgentResult("m", "", {"correction": 2.0, "meal": 3.0, "rules_fired": []})}
    assert {"A1", "A2"} <= {f.rule for f in audit(ctx(readings=flat(60)), results)}
    assert results["medication"].details["correction"] == 0

def test_auditor_a3_stale():
    results = {"medication": AgentResult("m", "", {"correction": 2.0, "meal": 0.0})}
    assert "A3" in {f.rule for f in audit(ctx(readings=flat(200), reading_age_min=40), results)}
    assert results["medication"].details["correction"] == 0

def test_auditor_a4_caps():
    results = {"medication": AgentResult("m", "", {"correction": 9.0, "meal": 11.0})}
    c = ctx(readings=flat(200), planned_carbs=100)
    assert "A4" in {f.rule for f in audit(c, results)}
    d = results["medication"].details
    assert d["correction"] <= 4 and d["correction"] + d["meal"] <= 12

def test_gate_conflict_removed_by_a5():
    results = {"nutrition": AgentResult("n", "", {}, ["Đi bộ sau ăn"])}
    audit(ctx(T1(), readings=flat(265), code="T1"), results)
    assert not results["nutrition"].actions

def test_auditor_a6_adds_critical():
    results = {"glucose": AgentResult("g", "")}
    assert "A6" in {f.rule for f in audit(ctx(readings=flat(48)), results)}
    assert results["glucose"].alerts[0].level == "critical"

def test_auditor_a7_emergency_no_dose():
    c = ctx(T1(), readings=flat(200)); c.shared["review_only"] = True
    results = {"medication": AgentResult("m", "", {"correction": 1.0, "meal": 0.0})}
    assert "A7" in {f.rule for f in audit(c, results)}

def test_random_profiles_never_violate():
    rng = np.random.default_rng(7)
    o = orch()
    for _ in range(250):
        on_ins = bool(rng.random() < .6)
        p = PatientProfile(age=float(rng.uniform(18, 90)), weight_kg=float(rng.uniform(40, 120)), height_cm=float(rng.uniform(145, 190)),
                           hba1c=float(rng.uniform(4.5, 12)), egfr=float(rng.uniform(10, 120)), on_insulin=on_ins,
                           medications=list(rng.choice(["metformin", "gliclazide", "pioglitazone", "sitagliptin"], int(rng.integers(0, 3)), replace=False)),
                           heart_failure=bool(rng.random() < .2), isf=float(rng.uniform(15, 100)), icr=float(rng.uniform(5, 25)),
                           diabetes_type_known=str(rng.choice(["T1", "T2"])) if rng.random() < .5 else None)
        n = int(rng.integers(1, 30))
        readings = list(np.clip(rng.uniform(40, 450) + np.cumsum(rng.normal(0, 8, n)), 36, 590))
        carbs = None if rng.random() < .4 else float(rng.uniform(0, 150))
        mins = None if rng.random() < .5 else float(rng.uniform(0, 300))
        last = None if rng.random() < .5 else float(rng.uniform(0, 8))
        rep = o.analyze(p, readings, planned_carbs=carbs, minutes_since_insulin=mins, last_insulin_units=last)
        assert rep.audit_findings == [], (p, readings, carbs, mins, last)
        m = rep.results.get("medication")
        if m and (m.details["correction"] or m.details["meal"]):
            assert p.on_insulin and readings[-1] >= 70

def test_every_report_has_disclaimer_and_sources():
    r = run("1"); assert "không thay thế bác sĩ" in r.disclaimer and r.sources
