"""Xuất KB từ Neo4j ra file và ảnh. Phần cần máy chủ thật tự bỏ qua khi chưa đặt NEO4J_PASSWORD."""
import os
import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from diabetes_mas.kb.export_kb import HEADER, draw, export_cypher, literal, read_graph, restore
from diabetes_mas.kb.neo4j_kb import load_neo4j, seed
from diabetes_mas.kb.store import load_kb

SMALL = {
    "nodes": [{"id": "a", "label": "Drug", "props": {"name": "metformin"}}, {"id": "b", "label": "Condition", "props": {"name": "egfr_lt_30"}},
              {"id": "c", "label": "Drug", "props": {"name": "linagliptin"}}, {"id": "r", "label": "Recommendation", "props": {"title": "Hạ đường huyết"}},
              {"id": "s", "label": "Source", "props": {"name": "WHO"}}],
    "edges": [{"from": "a", "to": "b", "type": "CONTRAINDICATED_IN", "props": {}}, {"from": "a", "to": "c", "type": "SUBSTITUTE_WHEN", "props": {}},
              {"from": "r", "to": "s", "type": "FROM", "props": {}}],
}


def test_literal_escapes_values():
    assert literal('a"b\\c\nd') == r'"a\"b\\c\nd"'
    assert literal(True) == "true" and literal(3) == "3" and literal(["x", "y"]) == '["x", "y"]'
    assert literal("Hạ đường") == '"Hạ đường"'

def test_draw_writes_png_and_svg(tmp_path):
    png, svg = draw(SMALL, tmp_path)
    assert png.read_bytes()[:4] == b"\x89PNG" and png.stat().st_size > 5000
    text = svg.read_text(encoding="utf-8")
    assert "metformin" in text and "Hạ đường huyết" in text and "Chống chỉ định" in text

def test_empty_graph_refuses_to_export(tmp_path):
    class Empty:
        def execute_query(self, *a, **k):
            return [], None, None
    keep = tmp_path / "kb.cypher"
    keep.write_text("cũ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="neo4j_kb"):
        export_cypher(Empty(), keep)
    assert keep.read_text(encoding="utf-8") == "cũ"

def test_restore_rejects_other_files(tmp_path):
    bad = tmp_path / "x.cypher"
    bad.write_text("MATCH (n) DETACH DELETE n;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không phải"):
        restore(None, bad)


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
    seed(drv, load_kb())
    drv.close()


def counts(driver):
    rows, _, _ = driver.execute_query("MATCH (n) WHERE n._kid IS NULL AND NOT n:KhongLienQuan RETURN count(n) AS n")
    rels, _, _ = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS n")
    return rows[0]["n"], rels[0]["n"]

def test_literal_is_valid_cypher(driver):
    v = 'a"b\\c\nd; đường'
    rows, _, _ = driver.execute_query(f"RETURN {literal(v)} AS v")
    assert rows[0]["v"] == v

def test_round_trip_rebuilds_same_kb(driver, tmp_path):
    path = tmp_path / "kb.cypher"
    export_cypher(driver, path)
    before = counts(driver)
    driver.execute_query("MATCH (d:Drug) DETACH DELETE d")
    assert counts(driver) != before
    restore(driver, path)
    assert counts(driver) == before
    kb, mem = load_neo4j(driver), load_kb()
    assert (kb.recommendations, kb.foods, kb.drugs, kb.drug_interactions, kb.substitutes) == \
           (mem.recommendations, mem.foods, mem.drugs, mem.drug_interactions, mem.substitutes)
    leftover, _, _ = driver.execute_query("MATCH (n) WHERE n._kid IS NOT NULL RETURN count(n) AS n")
    assert leftover[0]["n"] == 0

def test_export_is_stable(driver, tmp_path):
    a, b = tmp_path / "a.cypher", tmp_path / "b.cypher"
    export_cypher(driver, a); export_cypher(driver, b)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")

def test_restore_keeps_foreign_nodes(driver, tmp_path):
    path = tmp_path / "kb.cypher"
    export_cypher(driver, path)
    driver.execute_query("CREATE (:KhongLienQuan {name: 'giu-lai'})")
    try:
        restore(driver, path)
        rows, _, _ = driver.execute_query("MATCH (x:KhongLienQuan) RETURN count(x) AS n")
        assert rows[0]["n"] == 1
    finally:
        driver.execute_query("MATCH (x:KhongLienQuan) DETACH DELETE x")
    assert "KhongLienQuan" not in path.read_text(encoding="utf-8")

def test_broken_file_leaves_graph_untouched(driver, tmp_path):
    before = counts(driver)
    bad = tmp_path / "bad.cypher"
    bad.write_text(HEADER + "\nMATCH (n:Drug) DETACH DELETE n;\nTHIS IS NOT CYPHER;\n", encoding="utf-8")
    with pytest.raises(Neo4jError):
        restore(driver, bad)
    assert counts(driver) == before

def test_image_from_real_graph(driver, tmp_path):
    g = read_graph(driver)
    png, svg = draw(g, tmp_path)
    assert png.exists() and "pioglitazone" in svg.read_text(encoding="utf-8")
