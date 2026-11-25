"""KB trên đồ thị Neo4j. Nạp dữ liệu: python -m diabetes_mas.kb.neo4j_kb

Bản trong bộ nhớ (store.py) là nguồn dữ liệu gốc và là dự phòng khi không có Neo4j.
BM25 của RAG vẫn chạy trong tiến trình, trên các khuyến nghị đọc từ đồ thị.
"""
import logging
import os
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from .clinical import demo_patients, load_clinical
from .store import KnowledgeBase, load_kb

log = logging.getLogger(__name__)
LABELS = ["Recommendation", "Source", "Topic", "Food", "FoodCategory", "Drug", "Condition",
          "Disease", "Symptom", "RiskFactor", "Measurement", "Patient"]
Q_INTERACTIONS = """
MATCH (a:Drug)-[i:INTERACTS_WITH]->(b:Drug)
WHERE a.name IN $names AND b.name IN $names
RETURN a.name AS a, b.name AS b, i.severity AS severity, i.message AS message ORDER BY i.seq"""
Q_ALTERNATIVES = """
MATCH (d:Drug {name: $drug})-[s:SUBSTITUTE_WHEN {condition: $condition}]->(alt:Drug)
RETURN d.name AS drug, s.condition AS condition, alt.name AS alternative, s.why AS why ORDER BY s.seq"""


def _rows(driver, cypher: str, **params) -> list[dict]:
    records, _, _ = driver.execute_query(cypher, parameters_=params)
    return [r.data() for r in records]


def _seed_tx(tx, kb: KnowledgeBase) -> None:
    tx.run("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) DETACH DELETE n", labels=LABELS)
    tx.run("""UNWIND $rows AS r
              MERGE (s:Source {name: r.source}) SET s.url = r.url MERGE (t:Topic {name: r.topic})
              CREATE (n:Recommendation {seq: r.seq, title: r.title, text: r.text})
              CREATE (n)-[:FROM]->(s) CREATE (n)-[:ABOUT]->(t)""",
           rows=[{**r, "seq": i} for i, r in enumerate(kb.recommendations)])
    tx.run("""UNWIND $rows AS r
              MERGE (c:FoodCategory {name: r.category})
              CREATE (f:Food {seq: r.seq, name: r.name, carb_g: r.carb_g, portion: r.portion, gi: r.gi, meals: r.meals,
                              allergens: r.allergens, high_sodium: r.high_sodium, vegetarian: r.vegetarian})
              CREATE (f)-[:IN_CATEGORY]->(c)""",
           rows=[{**f, "seq": i} for i, f in enumerate(kb.foods)])
    tx.run("""UNWIND $rows AS r
              CREATE (:Drug {seq: r.seq, name: r.name, aliases: r.aliases, not_for_types: r.not_for_types, url: r.url})""",
           rows=[{**d, "seq": i} for i, d in enumerate(kb.drugs)])
    for rel, key in (("CONTRAINDICATED_IN", "contraindications"), ("CAUTION_IN", "cautions")):
        rows = [{"drug": d["name"], "code": c, "seq": i, "note": d["notes"].get(c)}
                for d in kb.drugs for i, c in enumerate(d[key])]
        tx.run(f"""UNWIND $rows AS r
                   MATCH (d:Drug {{name: r.drug}}) MERGE (c:Condition {{name: r.code}})
                   CREATE (d)-[:{rel} {{seq: r.seq, note: r.note}}]->(c)""", rows=rows)
    tx.run("""UNWIND $rows AS r
              MATCH (a:Drug {name: r.a}), (b:Drug {name: r.b})
              CREATE (a)-[:INTERACTS_WITH {seq: r.seq, severity: r.severity, message: r.message}]->(b)""",
           rows=[{**x, "seq": i} for i, x in enumerate(kb.drug_interactions)])
    tx.run("""UNWIND $rows AS r
              MATCH (d:Drug {name: r.drug}), (alt:Drug {name: r.alternative})
              CREATE (d)-[:SUBSTITUTE_WHEN {seq: r.seq, condition: r.condition, why: r.why}]->(alt)""",
           rows=[{**x, "seq": i} for i, x in enumerate(kb.substitutes)])
    _seed_clinical(tx, kb)


def _seed_clinical(tx, kb: KnowledgeBase) -> None:
    """Bệnh, triệu chứng, yếu tố nguy cơ, chỉ số đo (clinical.json) và bệnh nhân mẫu (scenarios.py)."""
    c = load_clinical()
    tx.run("UNWIND $rows AS r CREATE (:Disease {seq: r.seq, code: r.code, name: r.name})",
           rows=[{"seq": i, "code": d["code"], "name": d["name"]} for i, d in enumerate(c["diseases"])])
    tx.run("""UNWIND $rows AS r MATCH (a:Disease {code: r.child}), (b:Disease {code: r.parent})
              CREATE (a)-[:SUBTYPE_OF]->(b)""",
           rows=[{"child": d["code"], "parent": d["subtype_of"]} for d in c["diseases"] if d.get("subtype_of")])
    tx.run("""UNWIND $rows AS r MATCH (a:Disease {code: r.src}), (b:Disease {code: r.dst})
              CREATE (a)-[:MAY_LEAD_TO {seq: r.seq, note: r.note}]->(b)""",
           rows=[{"src": e["from"], "dst": e["to"], "note": e["note"], "seq": i} for i, e in enumerate(c["leads_to"])])
    tx.run("""UNWIND $rows AS r MATCH (d:Disease {code: r.code}), (n:Recommendation {title: r.title})
              CREATE (d)-[:HAS_RECOMMENDATION {seq: r.seq}]->(n)""",
           rows=[{"code": d["code"], "title": t, "seq": j} for d in c["diseases"] for j, t in enumerate(d["recommendations"])])
    tx.run("""UNWIND $rows AS r MATCH (d:Disease {code: r.disease})
              MERGE (s:Symptom {name: r.name}) ON CREATE SET s.seq = r.seq
              CREATE (d)-[:HAS_SYMPTOM {severity: r.severity, seq: r.seq}]->(s)""",
           rows=[{"disease": g["disease"], "severity": g["severity"], "name": n, "seq": i}
                 for i, (g, n) in enumerate((g, n) for g in c["symptoms"] for n in g["names"])])
    tx.run("""UNWIND $rows AS r CREATE (rf:RiskFactor {seq: r.seq, code: r.code, name: r.name})
              WITH rf, r UNWIND r.increases AS code MATCH (d:Disease {code: code})
              CREATE (rf)-[:INCREASES_RISK_OF]->(d)""",
           rows=[{**r, "seq": i} for i, r in enumerate(c["risk_factors"])])
    tx.run("UNWIND $rows AS r CREATE (:Measurement {seq: r.seq, code: r.code, name: r.name, unit: r.unit})",
           rows=[{**m, "seq": i} for i, m in enumerate(c["measurements"])])
    tx.run("""UNWIND $rows AS r MATCH (m:Measurement {code: r.measurement}), (d:Disease {code: r.disease})
              CREATE (m)-[:INDICATES {seq: r.seq, low: r.low, high: r.high, rule: r.rule}]->(d)""",
           rows=[{**x, "seq": i} for i, x in enumerate(c["indicates"])])
    patients = demo_patients(kb)
    tx.run("UNWIND $rows AS r CREATE (:Patient {seq: r.seq, name: r.name, title: r.title, age: r.age, sex: r.sex, demo: true})",
           rows=[{k: p[k] for k in ("name", "title", "age", "sex")} | {"seq": i} for i, p in enumerate(patients)])
    tx.run("""UNWIND $rows AS r MATCH (p:Patient {name: r.patient}), (m:Measurement {code: r.code})
              CREATE (p)-[:HAS_MEASUREMENT {value: r.value}]->(m)""",
           rows=[{"patient": p["name"], **m} for p in patients for m in p["measurements"]])
    tx.run("""UNWIND $rows AS r MATCH (p:Patient {name: r.patient}), (rf:RiskFactor {code: r.code})
              CREATE (p)-[:HAS_RISK_FACTOR]->(rf)""",
           rows=[{"patient": p["name"], "code": c} for p in patients for c in p["risk_factors"]])
    tx.run("""UNWIND $rows AS r MATCH (p:Patient {name: r.patient}) MERGE (c:Condition {name: r.code})
              CREATE (p)-[:HAS_CONDITION]->(c)""",
           rows=[{"patient": p["name"], "code": c} for p in patients for c in p["conditions"]])
    tx.run("""UNWIND $rows AS r MATCH (p:Patient {name: r.patient}), (d:Drug {name: r.drug})
              CREATE (p)-[:TAKES]->(d)""",
           rows=[{"patient": p["name"], "drug": d} for p in patients for d in p["drugs"]])
    tx.run("""UNWIND $rows AS r MATCH (p:Patient {name: r.patient}), (d:Disease {code: r.code})
              CREATE (p)-[:DIAGNOSED_WITH {source: "đã chẩn đoán (ca mẫu)"}]->(d)""",
           rows=[{"patient": p["name"], "code": p["diagnosis"]} for p in patients if p["diagnosis"]])


def seed(driver, kb: KnowledgeBase) -> None:
    """Nạp lại trong một giao dịch; chỉ xoá các nhãn của KB, không đụng dữ liệu khác trong cơ sở dữ liệu."""
    with driver.session() as session:
        session.execute_write(_seed_tx, kb)


@dataclass
class Neo4jKB(KnowledgeBase):
    driver: Any = None

    def _query(self, cypher: str, **params) -> list[dict] | None:
        """None nghĩa là dùng bản đã nạp. Lỗi một lần thì chuyển hẳn sang bản đó, để không chờ timeout mỗi lần gọi."""
        if self.driver is None:
            return None
        try:
            return _rows(self.driver, cypher, **params)
        except (DriverError, Neo4jError) as e:
            log.warning("Neo4j lỗi, dùng bản đã nạp đến khi khởi động lại: %s", e)
            self.driver = None
            self.meta["backend"] = "bộ nhớ (Neo4j mất kết nối)"
            return None

    def interactions(self, names: list[str]) -> list[dict]:
        rows = self._query(Q_INTERACTIONS, names=list(names))
        return super().interactions(names) if rows is None else rows

    def alternatives(self, drug: str, condition: str) -> list[dict]:
        rows = self._query(Q_ALTERNATIVES, drug=drug, condition=condition)
        return super().alternatives(drug, condition) if rows is None else rows


def load_neo4j(driver) -> Neo4jKB:
    recs = _rows(driver, """MATCH (n:Recommendation)-[:FROM]->(s:Source), (n)-[:ABOUT]->(t:Topic)
                            RETURN n.title AS title, s.name AS source, s.url AS url, t.name AS topic, n.text AS text ORDER BY n.seq""")
    if not recs:
        raise RuntimeError("Đồ thị chưa có dữ liệu. Hãy chạy: python -m diabetes_mas.kb.neo4j_kb")
    foods = [r["food"] for r in _rows(driver, """
        MATCH (f:Food)-[:IN_CATEGORY]->(c:FoodCategory)
        RETURN f {.name, .carb_g, .portion, .gi, .meals, .allergens, .high_sodium, .vegetarian, category: c.name} AS food
        ORDER BY f.seq""")]
    drugs = []
    for row in _rows(driver, """
            MATCH (d:Drug)
            OPTIONAL MATCH (d)-[r:CONTRAINDICATED_IN|CAUTION_IN]->(c:Condition)
            WITH d, r, c ORDER BY r.seq
            WITH d, collect({rel: type(r), code: c.name, note: r.note}) AS rels
            RETURN d.name AS name, d.aliases AS aliases, d.not_for_types AS not_for_types, d.url AS url, rels ORDER BY d.seq"""):
        rels = [r for r in row["rels"] if r["code"]]
        drugs.append({"name": row["name"], "aliases": row["aliases"], "not_for_types": row["not_for_types"], "url": row["url"],
                      "contraindications": [r["code"] for r in rels if r["rel"] == "CONTRAINDICATED_IN"],
                      "cautions": [r["code"] for r in rels if r["rel"] == "CAUTION_IN"],
                      "notes": {r["code"]: r["note"] for r in rels if r["note"] is not None}})
    inter = _rows(driver, """MATCH (a:Drug)-[i:INTERACTS_WITH]->(b:Drug)
                             RETURN a.name AS a, b.name AS b, i.severity AS severity, i.message AS message ORDER BY i.seq""")
    subs = _rows(driver, """MATCH (d:Drug)-[s:SUBSTITUTE_WHEN]->(alt:Drug)
                            RETURN d.name AS drug, s.condition AS condition, alt.name AS alternative, s.why AS why ORDER BY s.seq""")
    return Neo4jKB(recs, foods, drugs, inter, subs, {**load_kb().meta, "backend": "neo4j"}, driver)


def _connect():
    pw = os.getenv("NEO4J_PASSWORD")
    if not pw:
        raise RuntimeError("Chưa đặt NEO4J_PASSWORD.")
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
                                  auth=(os.getenv("NEO4J_USER", "neo4j"), pw),
                                  connection_timeout=3, max_transaction_retry_time=3)
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        raise
    return driver


def get_kb() -> KnowledgeBase:
    """KB_BACKEND: auto (mặc định, Neo4j nếu dùng được, không thì bộ nhớ), neo4j (bắt buộc), memory."""
    mode = os.getenv("KB_BACKEND", "auto")
    if mode not in ("auto", "neo4j", "memory"):
        raise ValueError(f"KB_BACKEND không hợp lệ: {mode}")
    if mode == "memory" or (mode == "auto" and not os.getenv("NEO4J_PASSWORD")):
        return load_kb()
    try:
        driver = _connect()
        try:
            return load_neo4j(driver)
        except Exception:
            driver.close()
            raise
    except (DriverError, Neo4jError, RuntimeError) as e:
        if mode == "neo4j":
            raise
        log.warning("Không dùng được Neo4j (%s), chuyển sang KB trong bộ nhớ.", e)
        return load_kb()


if __name__ == "__main__":
    drv = _connect()
    data = load_kb()
    seed(drv, data)
    c = load_clinical()
    print(f"Đã nạp {len(data.recommendations)} khuyến nghị, {len(data.foods)} thực phẩm, {len(data.drugs)} thuốc, "
          f"{len(data.drug_interactions)} tương tác, {len(data.substitutes)} thuốc thay thế, {len(c['diseases'])} bệnh, "
          f"{sum(len(g['names']) for g in c['symptoms'])} triệu chứng, {len(c['risk_factors'])} yếu tố nguy cơ, "
          f"{len(c['measurements'])} loại chỉ số, {len(demo_patients(data))} bệnh nhân mẫu.")
    drv.close()
