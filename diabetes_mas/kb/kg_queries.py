"""Truy vấn nhiều bước nhảy trên Knowledge Graph, mỗi truy vấn có bản Cypher (Neo4j) và bản Python tương đương
(dùng khi không chạy Neo4j, và để test hai bản cho cùng kết quả).

python -m diabetes_mas.kb.kg_queries                     chạy mọi truy vấn cho "Ca 5" và in kết quả
python -m diabetes_mas.kb.kg_queries --patient "Ca 7"    chọn bệnh nhân khác
python -m diabetes_mas.kb.kg_queries --cypher            chỉ in câu Cypher để chép vào Neo4j Browser
"""
import argparse
import json

from .clinical import demo_patients, load_clinical
from .store import KnowledgeBase

# ---------- Cypher ----------

DRUG_REVIEW = """
// Thuốc -> chống chỉ định/thận trọng -> bệnh lý nền CỦA BỆNH NHÂN -> thuốc thay thế an toàn với bệnh nhân
MATCH (p:Patient {name: $patient})-[:TAKES]->(d:Drug)-[ci:CONTRAINDICATED_IN|CAUTION_IN]->(c:Condition)<-[:HAS_CONDITION]-(p)
OPTIONAL MATCH (d)-[s:SUBSTITUTE_WHEN {condition: c.name}]->(alt:Drug)
  WHERE NOT EXISTS { MATCH (alt)-[:CONTRAINDICATED_IN|CAUTION_IN]->(:Condition)<-[:HAS_CONDITION]-(p) }
    AND NOT EXISTS { MATCH (p)-[:TAKES]->(alt) }
RETURN d.name AS drug, type(ci) AS relation, c.name AS condition, ci.note AS note,
       alt.name AS alternative, s.why AS why
ORDER BY drug, relation, condition, alternative"""

SUGGESTED_DISEASES = """
// Bệnh nhân -> chỉ số đo -> ngưỡng -> bệnh gợi ý (-> bệnh cha)
MATCH (p:Patient {name: $patient})-[m:HAS_MEASUREMENT]->(ms:Measurement)-[r:INDICATES]->(d:Disease)
WHERE (r.low IS NULL OR m.value >= r.low) AND (r.high IS NULL OR m.value < r.high)
OPTIONAL MATCH (d)-[:SUBTYPE_OF]->(parent:Disease)
RETURN ms.name AS measurement, m.value AS value, ms.unit AS unit, r.rule AS rule, d.name AS disease, parent.name AS parent
ORDER BY measurement, disease"""

RISK_PROFILE = """
// Bệnh nhân -> yếu tố nguy cơ -> bệnh có nguy cơ tăng
MATCH (p:Patient {name: $patient})-[:HAS_RISK_FACTOR]->(rf:RiskFactor)-[:INCREASES_RISK_OF]->(d:Disease)
WITH d, rf ORDER BY rf.name
RETURN d.name AS disease, collect(rf.name) AS risk_factors, count(rf) AS n
ORDER BY n DESC, disease"""

COMPLICATION_ADVICE = """
// Bệnh nhân -> chẩn đoán -> (bệnh cha) -> biến chứng -> khuyến nghị -> nguồn  (4-5 bước nhảy)
MATCH (p:Patient {name: $patient})-[:DIAGNOSED_WITH]->(dx:Disease)-[:SUBTYPE_OF*0..1]->(g:Disease)-[:MAY_LEAD_TO]->(c:Disease)
WITH DISTINCT c
MATCH (c)-[h:HAS_RECOMMENDATION]->(r:Recommendation)-[:FROM]->(s:Source)
WITH c, h, r, s ORDER BY h.seq
RETURN c.name AS complication, collect(r.title) AS recommendations, collect(DISTINCT s.url) AS sources
ORDER BY complication"""

SYMPTOM_ADVICE = """
// Triệu chứng -> bệnh -> khuyến nghị
MATCH (sy:Symptom {name: $symptom})<-[h:HAS_SYMPTOM]-(d:Disease)-[k:HAS_RECOMMENDATION]->(r:Recommendation)
WITH d, h, k, r ORDER BY k.seq
RETURN d.name AS disease, h.severity AS severity, collect(r.title) AS recommendations
ORDER BY disease"""

QUERIES = {
    "drug_review": ("Thuốc đang dùng chống chỉ định với bệnh lý nền nào, thay bằng thuốc gì", DRUG_REVIEW),
    "suggested_diseases": ("Chỉ số đo gợi ý bệnh gì theo ngưỡng", SUGGESTED_DISEASES),
    "risk_profile": ("Yếu tố nguy cơ làm tăng nguy cơ bệnh nào", RISK_PROFILE),
    "complication_advice": ("Biến chứng cần phòng và khuyến nghị liên quan", COMPLICATION_ADVICE),
    "symptom_advice": ("Có triệu chứng thì nghĩ tới bệnh gì, nên làm gì", SYMPTOM_ADVICE),
}

# Câu 22: tạo Patient, Disease, quan hệ rồi truy vấn lại (chạy tay trong Neo4j Browser)
CREATE_EXAMPLE = """
MERGE (p:Patient {name: 'Bệnh nhân demo'}) SET p.age = 52, p.sex = 'F'
MERGE (d:Disease {code: 'T2'}) ON CREATE SET d.name = 'Đái tháo đường type 2'
MERGE (p)-[r:DIAGNOSED_WITH]->(d) SET r.since = 2020, r.source = 'bác sĩ'
RETURN p, r, d;

MATCH (p:Patient {name: 'Bệnh nhân demo'})-[r:DIAGNOSED_WITH]->(d:Disease)
RETURN p.name AS patient, p.age AS age, d.name AS disease, r.since AS since;"""


# ---------- Bản Python tương đương (không cần Neo4j) ----------

def _patient(patients, name):
    p = next((x for x in patients if x["name"] == name), None)
    if p is None:
        raise KeyError(f"Không có bệnh nhân '{name}'")
    return p


def _recs_of(kb, disease) -> list[dict]:
    by_title = {r["title"]: r for r in kb.recommendations}
    return [by_title[t] for t in disease["recommendations"] if t in by_title]


def memory_query(name: str, kb: KnowledgeBase, patient: str | None = None, symptom: str | None = None) -> list[dict]:
    c, patients = load_clinical(), demo_patients(kb)
    diseases = {d["code"]: d for d in c["diseases"]}
    meas = {m["code"]: m for m in c["measurements"]}
    if name == "drug_review":
        p = _patient(patients, patient)
        drugs = {d["name"]: d for d in kb.drugs}
        rows = []
        for dn in p["drugs"]:
            d = drugs[dn]
            for rel, key in (("CONTRAINDICATED_IN", "contraindications"), ("CAUTION_IN", "cautions")):
                for cond in d[key]:
                    if cond not in p["conditions"]:
                        continue
                    alts = [s for s in kb.substitutes if s["drug"] == dn and s["condition"] == cond
                            and not set(drugs[s["alternative"]]["contraindications"] + drugs[s["alternative"]]["cautions"]) & set(p["conditions"])
                            and s["alternative"] not in p["drugs"]]
                    base = {"drug": dn, "relation": rel, "condition": cond, "note": d["notes"].get(cond)}
                    rows += [{**base, "alternative": s["alternative"], "why": s["why"]} for s in alts] or [{**base, "alternative": None, "why": None}]
        return sorted(rows, key=lambda r: (r["drug"], r["relation"], r["condition"], r["alternative"] or ""))
    if name == "suggested_diseases":
        p = _patient(patients, patient)
        values = {m["code"]: m["value"] for m in p["measurements"]}
        rows = []
        for ind in c["indicates"]:
            v = values.get(ind["measurement"])
            if v is None or (ind["low"] is not None and v < ind["low"]) or (ind["high"] is not None and v >= ind["high"]):
                continue
            d, m = diseases[ind["disease"]], meas[ind["measurement"]]
            parent = diseases[d["subtype_of"]]["name"] if d.get("subtype_of") else None
            rows.append({"measurement": m["name"], "value": v, "unit": m["unit"], "rule": ind["rule"], "disease": d["name"], "parent": parent})
        return sorted(rows, key=lambda r: (r["measurement"], r["disease"]))
    if name == "risk_profile":
        p = _patient(patients, patient)
        rf = {r["code"]: r for r in c["risk_factors"]}
        acc = {}
        for code in p["risk_factors"]:
            for d in rf[code]["increases"]:
                acc.setdefault(diseases[d]["name"], []).append(rf[code]["name"])
        rows = [{"disease": d, "risk_factors": sorted(v), "n": len(v)} for d, v in acc.items()]
        return sorted(rows, key=lambda r: (-r["n"], r["disease"]))
    if name == "complication_advice":
        p = _patient(patients, patient)
        if not p["diagnosis"]:
            return []
        start = {p["diagnosis"]} | ({diseases[p["diagnosis"]]["subtype_of"]} if diseases[p["diagnosis"]].get("subtype_of") else set())
        comps = {e["to"] for e in c["leads_to"] if e["from"] in start}
        rows = []
        for code in comps:
            recs = _recs_of(kb, diseases[code])
            if recs:
                rows.append({"complication": diseases[code]["name"], "recommendations": [r["title"] for r in recs],
                             "sources": list(dict.fromkeys(r["url"] for r in recs))})
        return sorted(rows, key=lambda r: r["complication"])
    if name == "symptom_advice":
        rows = []
        for group in c["symptoms"]:
            if symptom in group["names"]:
                d = diseases[group["disease"]]
                recs = [r["title"] for r in _recs_of(kb, d)]
                if recs:
                    rows.append({"disease": d["name"], "severity": group["severity"], "recommendations": recs})
        return sorted(rows, key=lambda r: r["disease"])
    raise KeyError(name)


def run_query(name: str, kb: KnowledgeBase, patient: str | None = None, symptom: str | None = None) -> tuple[list[dict], str]:
    """Chạy trên Neo4j nếu KB đang nối Neo4j, không thì bản Python. Trả (kết quả, nơi chạy)."""
    driver = getattr(kb, "driver", None)
    if driver is not None:
        from .neo4j_kb import _rows
        try:
            return _rows(driver, QUERIES[name][1], patient=patient, symptom=symptom), "neo4j"
        except Exception:  # Neo4j lỗi giữa chừng: vẫn trả lời bằng bản Python
            pass
    return memory_query(name, kb, patient, symptom), "bộ nhớ"


def patient_report(kb: KnowledgeBase, patient: str) -> dict:
    out, where = {}, set()
    for name, (desc, cypher) in QUERIES.items():
        if name == "symptom_advice":
            continue
        rows, w = run_query(name, kb, patient=patient)
        where.add(w)
        out[name] = {"description": desc, "cypher": cypher.strip(), "rows": rows}
    return {"patient": patient, "backend": ", ".join(sorted(where)), "queries": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="Ca 5")
    ap.add_argument("--symptom", default="Vã mồ hôi")
    ap.add_argument("--cypher", action="store_true")
    a = ap.parse_args()
    if a.cypher:
        for name, (desc, cy) in QUERIES.items():
            print(f"// ===== {name}: {desc}{cy}\n")
        print("// ===== Câu 22: tạo Patient, Disease và quan hệ" + CREATE_EXAMPLE)
    else:
        from .neo4j_kb import get_kb
        kb = get_kb()
        rep = patient_report(kb, a.patient)
        rows, w = run_query("symptom_advice", kb, symptom=a.symptom)
        rep["queries"]["symptom_advice"] = {"description": QUERIES["symptom_advice"][0], "symptom": a.symptom, "rows": rows}
        for q in rep["queries"].values():
            q.pop("cypher", None)
        print(f"Chạy trên: {rep['backend']}")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
