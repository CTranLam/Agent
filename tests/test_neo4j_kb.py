"""Chọn backend chạy không cần Neo4j; phần tích hợp chỉ chạy khi có NEO4J_PASSWORD và máy chủ thật."""
import os
import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.kb.neo4j_kb import Neo4jKB, get_kb, load_neo4j, seed
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.scenarios import SCENARIOS
from tests.fakes import FakePredictor

DEAD = "bolt://127.0.0.1:1"


def test_auto_without_password_uses_memory(monkeypatch):
    monkeypatch.delenv("KB_BACKEND", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert get_kb().meta["backend"] == "bộ nhớ"

def test_memory_backend_never_connects(monkeypatch):
    monkeypatch.setenv("KB_BACKEND", "memory")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("NEO4J_URI", DEAD)
    assert get_kb().meta["backend"] == "bộ nhớ"

def test_auto_with_dead_server_falls_back(monkeypatch):
    monkeypatch.delenv("KB_BACKEND", raising=False)
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("NEO4J_URI", DEAD)
    assert get_kb().meta["backend"] == "bộ nhớ"

def test_forced_neo4j_without_password_fails(monkeypatch):
    monkeypatch.setenv("KB_BACKEND", "neo4j")
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        get_kb()

def test_forced_neo4j_with_dead_server_fails(monkeypatch):
    monkeypatch.setenv("KB_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("NEO4J_URI", DEAD)
    with pytest.raises((DriverError, Neo4jError)):
        get_kb()

def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("KB_BACKEND", "redis")
    with pytest.raises(ValueError, match="KB_BACKEND"):
        get_kb()


@pytest.fixture(scope="module")
def driver():
    pw = os.getenv("NEO4J_PASSWORD")
    if not pw:
        pytest.skip("chưa đặt NEO4J_PASSWORD")
    drv = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"), auth=(os.getenv("NEO4J_USER", "neo4j"), pw))
    try:
        drv.verify_connectivity()
    except (DriverError, Neo4jError) as e:
        pytest.skip(f"không kết nối được Neo4j: {e}")
    seed(drv, load_kb())
    yield drv
    drv.close()


@pytest.fixture(scope="module")
def graph(driver):
    return load_neo4j(driver)


def test_graph_holds_same_knowledge_as_files(graph):
    mem = load_kb()
    assert graph.recommendations == mem.recommendations
    assert graph.foods == mem.foods
    assert graph.drugs == mem.drugs
    assert graph.drug_interactions == mem.drug_interactions
    assert graph.substitutes == mem.substitutes
    assert graph.meta["backend"] == "neo4j" and graph.meta["reviewed"] is False

def test_cypher_queries_match_memory(graph, driver):
    mem = load_kb()
    names = [d["name"] for d in mem.drugs]
    assert graph.interactions(names) == mem.interactions(names)
    assert graph.interactions(["gliclazide", "metformin"]) == []
    for s in mem.substitutes:
        assert graph.alternatives(s["drug"], s["condition"]) == mem.alternatives(s["drug"], s["condition"])
    assert graph.alternatives("metformin", "heart_failure") == []

def test_relationships_really_in_graph(driver):
    rows, _, _ = driver.execute_query(
        "MATCH (:Drug {name: 'pioglitazone'})-[:CONTRAINDICATED_IN]->(c:Condition) RETURN c.name AS c")
    assert [r["c"] for r in rows] == ["heart_failure"]
    rows, _, _ = driver.execute_query("MATCH (:Recommendation)-[:FROM]->(s:Source) RETURN count(DISTINCT s) AS n")
    assert rows[0]["n"] >= 3

@pytest.mark.parametrize("key", SCENARIOS)
def test_scenarios_identical_on_both_backends(graph, key):
    class Fixed:
        def suggest(self, g, t, i): return 2.0
    s = SCENARIOS[key]
    def run(kb):
        return Orchestrator(FakePredictor(s.get("prob", 0.01)), kb, RAG(kb), Fixed()).analyze(
            s["profile"], s["readings"], planned_carbs=s.get("planned_carbs"),
            minutes_since_insulin=s.get("minutes_since_insulin"), last_insulin_units=s.get("last_insulin_units")).to_dict()
    assert run(graph) == run(load_kb())

def test_seed_is_repeatable_and_keeps_foreign_nodes(driver, graph):
    driver.execute_query("CREATE (:KhongLienQuan {name: 'giu-lai'})")
    try:
        seed(driver, load_kb())
        n, _, _ = driver.execute_query("MATCH (d:Drug) RETURN count(d) AS n")
        assert n[0]["n"] == len(load_kb().drugs)
        kept, _, _ = driver.execute_query("MATCH (x:KhongLienQuan) RETURN count(x) AS n")
        assert kept[0]["n"] == 1
    finally:
        driver.execute_query("MATCH (x:KhongLienQuan) DETACH DELETE x")

def test_empty_graph_is_not_used(driver, monkeypatch):
    with pytest.raises(RuntimeError, match="neo4j_kb"):
        driver.execute_query("MATCH (r:Recommendation) DETACH DELETE r")
        try:
            load_neo4j(driver)
        finally:
            seed(driver, load_kb())

def test_query_failure_falls_back_to_loaded_copy(driver):
    pw = os.environ["NEO4J_PASSWORD"]
    d2 = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"), auth=(os.getenv("NEO4J_USER", "neo4j"), pw))
    kb = load_neo4j(d2)
    d2.close()
    assert isinstance(kb, Neo4jKB)
    assert kb.alternatives("pioglitazone", "heart_failure")
    assert kb.interactions(["gliclazide", "insulin aspart"])
    assert kb.driver is None and "mất kết nối" in kb.meta["backend"]  # không chờ timeout ở các lần sau


# ---- phần lâm sàng: Patient, Disease, Symptom, RiskFactor, Measurement ----

def test_clinical_nodes_in_graph(driver):
    from diabetes_mas.kb.clinical import demo_patients, load_clinical
    c = load_clinical()
    expect = {"Disease": len(c["diseases"]), "RiskFactor": len(c["risk_factors"]), "Measurement": len(c["measurements"]),
              "Symptom": len({n for g in c["symptoms"] for n in g["names"]}), "Patient": len(demo_patients(load_kb()))}
    for label, n in expect.items():
        rows, _, _ = driver.execute_query(f"MATCH (x:`{label}`) RETURN count(x) AS n")
        assert rows[0]["n"] == n, label

@pytest.mark.parametrize("patient", [f"Ca {i}" for i in range(1, 9)])
def test_multi_hop_cypher_matches_memory(graph, patient):
    from diabetes_mas.kb.kg_queries import QUERIES, memory_query, run_query
    for name in QUERIES:
        if name == "symptom_advice":
            continue
        rows, where = run_query(name, graph, patient=patient)
        assert where == "neo4j"
        assert rows == memory_query(name, load_kb(), patient=patient), name

def test_symptom_cypher_matches_memory(graph):
    from diabetes_mas.kb.clinical import load_clinical
    from diabetes_mas.kb.kg_queries import memory_query, run_query
    for s in {n for g in load_clinical()["symptoms"] for n in g["names"]}:
        rows, where = run_query("symptom_advice", graph, symptom=s)
        assert where == "neo4j" and rows == memory_query("symptom_advice", load_kb(), symptom=s), s
