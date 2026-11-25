"""Phân vai: RLAgent chỉ đề xuất, RuleAgent mới quyết định liều."""
import pytest
from diabetes_mas.agents.medication_agent import MedicationAgent
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.agents.rl_agent import RLAgent
from diabetes_mas.agents.rule_agent import RuleAgent
from diabetes_mas.schemas import PatientProfile
from diabetes_mas.scenarios import SCENARIOS
from tests.fakes import FakePredictor
from tests.test_agents import Fixed, T1, ctx, flat, kb, rag


def case(g=205, p=None, **kw):
    return ctx(p or T1(), readings=flat(g), code="T1", minutes_since_insulin=300, **kw)

def test_rl_agent_only_proposes_from_its_state():
    seen = []
    class Spy:
        def suggest(self, *state): seen.append(state); return 1.5
    c = case()
    assert RLAgent(Spy()).propose(c) == 1.5
    assert seen == [(205.0, c.forecast["trend_per_5min"], 40)]

def test_rl_agent_stays_silent_without_isf():
    assert RLAgent(Fixed(4.0)).propose(case(p=PatientProfile(on_insulin=True))) == 0.0

def test_rule_agent_vetoes_any_rl_proposal_when_low():
    for proposed in (0.5, 4.0, 100.0):
        d = RuleAgent().decide(case(60), proposed)
        assert d.correction == 0 and d.blocked_reason.startswith("R2")

def test_rule_agent_reduces_proposal_but_never_raises_it():
    assert RuleAgent().decide(case(), 100.0).correction < 100
    assert RuleAgent().decide(case(), 1.0).correction <= 1.0
    assert RuleAgent().decide(case(), 0.0).correction == 0

def test_medication_agent_reports_both_roles():
    d = MedicationAgent(kb, rag, Fixed()).run(case()).details
    assert d["proposed"] == 2.0 and d["rl"] == {"glucose": 205.0, "trend_per_5min": 0.0, "isf": 40}

def test_blocked_case_hides_rl_output():
    d = MedicationAgent(kb, rag, Fixed()).run(case(60)).details
    assert d["proposed"] is None and d["rl"] is None and d["blocked_reason"]

def test_emergency_runs_neither_agent():
    class Boom:
        def suggest(self, *a): raise AssertionError("RL không được chạy khi khẩn cấp")
    c = case(48); c.shared["review_only"] = True
    d = MedicationAgent(kb, rag, Boom()).run(c).details
    assert d["rl"] is None and d["correction"] is None

@pytest.mark.parametrize("key,kind", [("diagnosis", "Deep learning + Rule"), ("glucose", "Deep learning + Rule"), ("nutrition", "Rule + KB"),
                                      ("medication", "RL + Rule + KB"), ("exercise", "Rule + KB")])
def test_every_agent_states_its_kind(key, kind):
    s = SCENARIOS["4"]
    rep = Orchestrator(FakePredictor(), kb, rag, Fixed()).analyze(s["profile"], flat(150))
    assert rep.results[key].kind == kind

def test_agent_names_match_class_names():
    agents = [*Orchestrator(FakePredictor(), kb, rag, Fixed()).agents.values(), RLAgent(Fixed()), RuleAgent()]
    assert len(agents) == 7 and all(a.name == type(a).__name__ for a in agents)
