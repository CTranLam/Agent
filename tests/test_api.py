import pytest
from fastapi.testclient import TestClient
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.api import create_app
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from tests.fakes import FakePredictor

kb = load_kb(); rag = RAG(kb)


class Fixed:
    def suggest(self, g, t, i): return 2.0

def client():
    return TestClient(create_app(Orchestrator(FakePredictor(), kb, rag, Fixed()), rag))

def body(**kw):
    return {"profile": {}, "readings": [120.0] * 12} | kw


def test_analyze_ok():
    r = client().post("/analyze", json=body(readings=[100]))
    assert r.status_code == 200 and "disclaimer" in r.json()

def test_mmol_unit_converted():
    r = client().post("/analyze", json=body(readings=[5.6]*12, unit="mmol/L"))
    assert r.status_code == 200 and r.json()["results"]["glucose"]["details"]["current"] == pytest.approx(100.9, abs=.1)

@pytest.mark.parametrize("bad", [
    {"readings": []}, {"readings": [5.6, 5.8]}, {"readings": [120] * 300}, {"unit": "abc"},
    {"profile": {"weight_kg": 0}}, {"profile": {"height_cm": 0}}, {"profile": {"isf": 0}},
    {"profile": {"icr": 0}}, {"profile": {"age": -1}}, {"profile": {"diabetes_type_known": "X"}},
    {"planned_carbs": -5}, {"minutes_since_insulin": -1},
])
def test_bad_input_is_422(bad):  # Review Focus 5
    assert client().post("/analyze", json=body(**bad)).status_code == 422

def test_mmol_error_message_visible():
    r = client().post("/analyze", json=body(readings=[5.6, 5.8]))
    assert "mmol/L" in r.json()["detail"]

def test_nan_reading_is_422():
    raw = '{"profile": {}, "readings": [NaN]}'
    r = client().post("/analyze", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422

def test_ask_empty():
    assert "không tìm thấy" in client().post("/ask", json={"question": " "}).json()["answer"].lower()

def test_ask_returns_evidence():
    j = client().post("/ask", json={"question": "tụt đường phải làm gì"}).json()
    assert j["evidence"][0]["source"] and j["evidence"][0]["title"] in j["answer"]

def test_scenarios_and_index():
    c = client()
    scen = c.get("/scenarios").json()
    assert len(scen) == 8 and scen["4"]["planned_carbs"] == 60 and scen["4"]["profile"]["isf"] == 40
    assert "Phân tích" in c.get("/").text

@pytest.mark.parametrize("key", list("12345678"))
def test_every_scenario_round_trips(key):
    c = client()
    s = c.get("/scenarios").json()[key]
    payload = {k: s[k] for k in ("profile", "readings")} | {k: s.get(k) for k in ("planned_carbs", "minutes_since_insulin", "last_insulin_units")}
    assert c.post("/analyze", json=payload).status_code == 200

def test_health():
    j = client().get("/health").json()
    assert set(j) == {"kb", "rag", "retriever", "llm", "dl"} and j["rag"] == "KB" and j["llm"] == "tắt"

def test_graph():
    j = client().get("/graph").json()
    ids = {n["id"] for n in j["nodes"]}
    assert {"metformin", "egfr_lt_30"} <= ids
    assert {"contraindicated", "caution", "substitute", "interacts"} <= {e["kind"] for e in j["edges"]}
    assert all(e["from"] in ids and e["to"] in ids for e in j["edges"])

def test_no_external_llm_dependency():
    from pathlib import Path
    src = "".join(p.read_text(encoding="utf-8") for p in Path("diabetes_mas").rglob("*.py"))
    assert "anthropic" not in src.lower()
    assert "anthropic" not in Path("requirements.txt").read_text()


def test_kg_patients_and_multi_hop():
    c = client()
    j = c.get("/kg/patients").json()
    assert len(j["patients"]) == 8 and "Vã mồ hôi" in j["symptoms"]
    rep = c.get("/kg/patient/Ca 7").json()
    assert rep["backend"] == "bộ nhớ"
    assert rep["queries"]["drug_review"]["rows"][0]["alternative"] == "empagliflozin"
    assert "MATCH (p:Patient" in rep["queries"]["drug_review"]["cypher"]
    assert c.get("/kg/patient/Khong co").status_code == 404
    assert c.get("/kg/symptom/Vã mồ hôi").json()["rows"][0]["disease"] == "Hạ đường huyết"

def test_ask_can_force_extractive():
    j = client().post("/ask", json={"question": "tụt đường phải làm gì", "generate": False}).json()
    assert j["mode"] == "extractive" and j["evidence"]
