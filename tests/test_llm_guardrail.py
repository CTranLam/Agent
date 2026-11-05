from types import SimpleNamespace as NS
import anthropic
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.guardrail import is_grounded
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.llm import LLM
from diabetes_mas.schemas import PatientProfile
from tests.fakes import FakePredictor

kb = load_kb(); rag = RAG(kb)


class Fixed:
    def suggest(self, g, t, i): return 2.0

class FakeLLM:
    enabled = True; last_error = "boom"
    def __init__(self, text): self.text = text
    def complete(self, system, user): return self.text

def fake_client(fn): return NS(beta=NS(messages=NS(create=fn)))


def test_invented_dose_rejected():
    ok, bad = is_grounded("Hãy tiêm 9 đơn vị", "liều 2,0 đv")
    assert not ok and bad

def test_same_number_other_format_passes():
    assert is_grounded("tiêm 2.0 đv", "liều 2,0 đv")[0]
    assert is_grounded("tiêm 2 đv", "liều 2,0 đv")[0]

def test_plain_text_without_numbers_passes():
    assert is_grounded("Hãy đo lại đường huyết.", "gì đó")[0]

def test_llm_error_returns_none():
    def boom(**k): raise anthropic.APIConnectionError(request=None)
    llm = LLM(client=fake_client(boom))
    assert llm.complete("s", "u") is None and llm.last_error

def test_llm_refusal_returns_none():
    llm = LLM(client=fake_client(lambda **k: NS(stop_reason="refusal", content=[])))
    assert llm.complete("s", "u") is None and "từ chối" in llm.last_error

def test_llm_reads_text_blocks():
    resp = NS(stop_reason="end_turn", content=[NS(type="thinking"), NS(type="text", text="xin chào")])
    assert LLM(client=fake_client(lambda **k: resp)).complete("s", "u") == "xin chào"

def test_answer_without_llm_is_extractive():
    assert rag.answer("tụt đường")["mode"] == "extractive"

def test_answer_drops_ungrounded_llm():
    a = rag.answer("tụt đường", llm=FakeLLM("tiêm 99 đơn vị"))
    assert a["mode"] == "extractive" and "99" not in a["answer"]

def test_answer_uses_grounded_llm():
    a = rag.answer("tụt đường", llm=FakeLLM("Dùng carbohydrate nhanh rồi đo lại."))
    assert a["mode"] == "llm" and a["evidence"]

def test_no_evidence_never_calls_llm():
    a = rag.answer("xyzzy qwerty", llm=FakeLLM("tiêm 2 đv"))
    assert a["mode"] == "extractive" and "không tìm thấy" in a["answer"].lower()

def test_synthesis_note_when_disabled():
    r = Orchestrator(FakePredictor(), kb, rag, Fixed()).analyze(PatientProfile(), [100.0])
    assert r.synthesis is None and "LLM" in r.synthesis_note

def test_synthesis_rejected_when_ungrounded():
    r = Orchestrator(FakePredictor(), kb, rag, Fixed(), llm=FakeLLM("tiêm 77 đơn vị")).analyze(PatientProfile(), [100.0])
    assert r.synthesis is None and "77" in r.synthesis_note

def test_synthesis_kept_when_grounded():
    r = Orchestrator(FakePredictor(), kb, rag, Fixed(), llm=FakeLLM("Tiếp tục duy trì.")).analyze(PatientProfile(), [100.0])
    assert r.synthesis == "Tiếp tục duy trì."
