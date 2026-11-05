import pytest
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb


def test_kb_minimum_content():
    kb = load_kb()
    assert len(kb.recommendations) >= 25 and len(kb.foods) >= 25 and len(kb.drugs) >= 9
    assert {"hypoglycemia", "hyperglycemia", "nutrition", "exercise", "medication", "insulin",
            "complications", "special_groups"} <= {r["topic"] for r in kb.recommendations}

def test_slang_finds_hypo():
    assert "hạ đường huyết" in RAG(load_kb()).retrieve("bị tụt đường phải làm gì")[0].title.lower()

def test_every_evidence_has_source():
    ev = RAG(load_kb()).retrieve("ăn gì tốt", k=5)
    assert ev and all(e.source for e in ev)

def test_empty_or_off_topic_returns_nothing():  # Review Focus 4
    r = RAG(load_kb())
    assert r.retrieve("") == [] and r.retrieve("   ") == [] and r.retrieve("xyzzy qwerty") == []
    assert "không tìm thấy" in r.extractive([]).lower()

@pytest.mark.parametrize("q", ["thời tiết hôm nay thế nào", "giá vàng hôm nay", "đội nào vô địch bóng đá"])
def test_everyday_off_topic_returns_nothing(q):
    assert RAG(load_kb()).retrieve(q) == []

def test_single_drug_word_still_found():
    assert RAG(load_kb()).retrieve("metformin")

def test_match_drug_with_dose():  # Review Focus 3
    kb = load_kb()
    assert kb.match_drug("Metformin 850mg")["name"].lower() == "metformin"
    assert kb.match_drug("thuoc la") is None

def test_alternatives():
    assert load_kb().alternatives("pioglitazone", "heart_failure")

def test_interactions_only_when_both_taken():
    kb = load_kb()
    assert kb.interactions(["gliclazide", "insulin aspart"])
    assert kb.interactions(["gliclazide", "metformin"]) == []

def test_kb_not_marked_reviewed():
    assert load_kb().meta["reviewed"] is False

@pytest.mark.parametrize("q", ["đếm carb để tính liều", "ăn mặn có hại không", "đường 300 có được tập không"])
def test_short_domain_words_not_filtered(q):
    assert RAG(load_kb()).retrieve(q)
