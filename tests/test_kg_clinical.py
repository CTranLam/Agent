"""Phần lâm sàng của Knowledge Graph và truy vấn nhiều bước (bản Python, không cần Neo4j)."""
import pytest
from diabetes_mas.kb.clinical import demo_patients, load_clinical
from diabetes_mas.kb.kg_queries import QUERIES, memory_query, patient_report, run_query
from diabetes_mas.kb.store import load_kb

KB = load_kb()
C = load_clinical()


def test_schema_references_are_consistent():
    codes = {d["code"] for d in C["diseases"]}
    titles = {r["title"] for r in KB.recommendations}
    for d in C["diseases"]:
        assert d.get("subtype_of") in (None, *codes)
        assert d["recommendations"] and set(d["recommendations"]) <= titles, d["code"]
    assert {e["from"] for e in C["leads_to"]} | {e["to"] for e in C["leads_to"]} <= codes
    assert {g["disease"] for g in C["symptoms"]} <= codes
    assert {d for r in C["risk_factors"] for d in r["increases"]} <= codes
    assert {i["measurement"] for i in C["indicates"]} <= {m["code"] for m in C["measurements"]}
    assert {i["disease"] for i in C["indicates"]} <= codes

def test_every_scenario_becomes_a_patient():
    names = [p["name"] for p in demo_patients(KB)]
    assert names == [f"Ca {i}" for i in range(1, 9)]

def test_patient_links_reuse_existing_kb_nodes():
    drugs = {d["name"] for d in KB.drugs}
    conds = {c for d in KB.drugs for c in d["contraindications"] + d["cautions"]}
    for p in demo_patients(KB):
        assert set(p["drugs"]) <= drugs and set(p["conditions"]) <= conds

def test_drug_review_multi_hop_ca7():
    rows = memory_query("drug_review", KB, patient="Ca 7")  # suy tim + pioglitazone
    assert rows == [{"drug": "pioglitazone", "relation": "CONTRAINDICATED_IN", "condition": "heart_failure",
                     "note": rows[0]["note"], "alternative": "empagliflozin", "why": "có lợi trên suy tim"}]

def test_drug_review_multi_hop_ca5():
    rows = memory_query("drug_review", KB, patient="Ca 5")  # 80 tuổi, eGFR 25, gliclazide
    assert {(r["condition"], r["alternative"]) for r in rows} == {("age_ge_75", "linagliptin"), ("egfr_lt_30", "linagliptin")}

def test_alternative_blocked_by_patient_condition():
    # thay thế bị loại nếu chính nó vướng bệnh lý nền của bệnh nhân
    kb = load_kb()
    kb.substitutes = [{"drug": "pioglitazone", "condition": "heart_failure", "alternative": "sitagliptin", "why": "thử"}]
    rows = memory_query("drug_review", kb, patient="Ca 7")
    assert rows and rows[0]["alternative"] is None  # sitagliptin thận trọng khi suy tim

def test_suggested_diseases_use_thresholds():
    rows = memory_query("suggested_diseases", KB, patient="Ca 5")
    assert {r["disease"] for r in rows} == {"Đái tháo đường", "Bệnh thận do đái tháo đường", "Hạ đường huyết"}
    assert memory_query("suggested_diseases", KB, patient="Ca 2")[0]["disease"] == "Tiền đái tháo đường"

def test_complications_follow_subtype_and_lead_to():
    rows = memory_query("complication_advice", KB, patient="Ca 4")  # T1 -> DKA, và qua DM -> biến chứng chung
    comps = {r["complication"] for r in rows}
    assert {"Nhiễm toan ceton", "Hạ đường huyết", "Bệnh thận do đái tháo đường"} <= comps
    assert "Tăng áp lực thẩm thấu" not in comps  # chỉ nối từ type 2
    assert all(r["sources"] for r in rows)

def test_undiagnosed_patient_has_no_complication_path():
    assert memory_query("complication_advice", KB, patient="Ca 1") == []

def test_symptom_to_recommendation():
    rows = memory_query("symptom_advice", KB, symptom="Vã mồ hôi")
    assert rows[0]["disease"] == "Hạ đường huyết"
    assert "Xử trí hạ đường huyết theo quy tắc 15-15" in rows[0]["recommendations"]

def test_risk_profile_counts():
    rows = memory_query("risk_profile", KB, patient="Ca 7")
    t2 = next(r for r in rows if r["disease"] == "Đái tháo đường type 2")
    assert t2["n"] == 3 and rows[0] == t2

def test_unknown_patient():
    with pytest.raises(KeyError):
        memory_query("drug_review", KB, patient="Không có")

def test_without_neo4j_runs_in_memory():
    rows, where = run_query("drug_review", KB, patient="Ca 7")
    assert where == "bộ nhớ" and rows

def test_report_carries_cypher():
    rep = patient_report(KB, "Ca 5")
    assert set(rep["queries"]) == set(QUERIES) - {"symptom_advice"}
    assert all("MATCH" in q["cypher"] for q in rep["queries"].values())
