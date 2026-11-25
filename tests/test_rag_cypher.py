"""RAG đọc tri thức trực tiếp từ file .cypher đã xuất, không cần kết nối Neo4j."""
import json
from pathlib import Path
import pytest
from diabetes_mas.kb.rag import RAG, build_rag, read_cypher_recommendations
from diabetes_mas.kb.store import CYPHER_HEADER, load_kb

FILE = Path("exports/kb_neo4j.cypher")


def write(tmp_path, *lines):
    p = tmp_path / "kb.cypher"
    p.write_text("\n".join([CYPHER_HEADER, *lines]) + "\n", encoding="utf-8")
    return p

def rec(kid, seq, title, text):
    return f"CREATE (:`Recommendation` {{`_kid`: {kid}, `seq`: {seq}, `text`: {json.dumps(text, ensure_ascii=False)}, `title`: {json.dumps(title, ensure_ascii=False)}}});"

SRC = 'CREATE (:`Source` {`_kid`: 5, `name`: "Vinmec - Bài", `url`: "https://www.vinmec.com/x"});'
TOP = 'CREATE (:`Topic` {`_kid`: 6, `name`: "insulin"});'
EDGES = ["MATCH (a {_kid: %d}), (b {_kid: 5}) CREATE (a)-[:`FROM` {}]->(b);", "MATCH (a {_kid: %d}), (b {_kid: 6}) CREATE (a)-[:`ABOUT` {}]->(b);"]


def test_parses_tricky_text_and_links(tmp_path):
    text = 'Có "ngoặc", dấu: {`x`: 1}; xuống\ndòng và CREATE (:`Drug`)'
    p = write(tmp_path, rec(0, 0, "Tiêu đề", text), SRC, TOP, EDGES[0] % 0, EDGES[1] % 0)
    assert read_cypher_recommendations(p) == [{"title": "Tiêu đề", "text": text, "source": "Vinmec - Bài",
                                               "url": "https://www.vinmec.com/x", "topic": "insulin"}]

def test_keeps_seq_order(tmp_path):
    lines = [rec(0, 1, "B", "b"), rec(1, 0, "A", "a"), SRC, TOP] + [e % k for k in (0, 1) for e in EDGES]
    assert [r["title"] for r in read_cypher_recommendations(write(tmp_path, *lines))] == ["A", "B"]

def test_rejects_other_file(tmp_path):
    p = tmp_path / "x.cypher"
    p.write_text("MATCH (n) RETURN n;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không phải"):
        read_cypher_recommendations(p)

def test_rejects_recommendation_without_source(tmp_path):
    with pytest.raises(ValueError, match="thiếu"):
        read_cypher_recommendations(write(tmp_path, rec(0, 0, "A", "a"), TOP, EDGES[1] % 0))

def test_rejects_file_without_recommendations(tmp_path):
    with pytest.raises(ValueError, match="không có khuyến nghị"):
        read_cypher_recommendations(write(tmp_path, SRC))

def test_build_rag_uses_file_and_reports_origin(tmp_path):
    p = write(tmp_path, rec(0, 0, "Hạ đường huyết riêng", "nội dung từ file cypher"), SRC, TOP, EDGES[0] % 0, EDGES[1] % 0)
    rag = build_rag(load_kb(), p)
    assert rag.origin == "file .cypher" and [d["title"] for d in rag.docs] == ["Hạ đường huyết riêng"]
    assert rag.retrieve("hạ đường huyết")[0].url == "https://www.vinmec.com/x"

def test_build_rag_without_file_falls_back(tmp_path):
    rag = build_rag(load_kb(), tmp_path / "khong_co.cypher")
    assert rag.origin == "KB" and len(rag.docs) == len(load_kb().recommendations)

def test_corrupt_file_is_not_silently_ignored(tmp_path):
    bad = tmp_path / "bad.cypher"
    bad.write_text("rác\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_rag(load_kb(), bad)

def test_exported_file_matches_kb_sources():
    if not FILE.exists():
        pytest.skip("chưa xuất file .cypher")
    key = lambda r: r["title"]
    assert sorted(read_cypher_recommendations(FILE), key=key) == sorted(load_kb().recommendations, key=key)

def test_chat_same_results_from_file_and_kb():
    if not FILE.exists():
        pytest.skip("chưa xuất file .cypher")
    kb = load_kb()
    a, b = build_rag(kb, FILE), RAG(kb)
    for q in ("tụt đường phải làm gì", "metformin suy thận", "ăn gì tốt"):
        assert [e.title for e in a.retrieve(q)] == [e.title for e in b.retrieve(q)]
