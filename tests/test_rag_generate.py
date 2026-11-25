"""Bước sinh (G) của RAG và bộ truy xuất vector. LLM được giả lập nên test không cần mạng."""
import pytest
from diabetes_mas.kb.generate import Generator, build_prompt, generator_from_env, verify
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from diabetes_mas.kb.vector import LSAEmbedder, VectorIndex, rrf

KB = load_kb()


class FakeLLM:
    model = "llm-gia"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user):
        self.prompts.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def rag_with(reply):
    llm = FakeLLM(reply)
    return RAG(KB, generator=Generator(llm)), llm


Q = "bị tụt đường phải làm gì"


def grounded_reply(rag):
    ev = rag.retrieve(Q)
    i = next(n for n, e in enumerate(ev) if "15-15" in e.title) + 1
    return f"Khi có triệu chứng, hãy ăn hoặc uống 15-20 gam carbohydrate tác dụng nhanh và kiểm tra đường huyết sau 15 phút [{i}]."


def test_default_is_extractive_without_llm():
    a = RAG(KB).answer(Q)
    assert a["mode"] == "extractive" and a["evidence"] and "fallback_reason" not in a

def test_asking_generation_without_llm_explains():
    a = RAG(KB).answer(Q, generate=True)
    assert a["mode"] == "extractive" and "LLM_BASE_URL" in a["fallback_reason"]

def test_grounded_answer_is_shown():
    rag, llm = rag_with("")
    llm.reply = grounded_reply(rag)
    a = rag.answer(Q)
    assert a["mode"] == "generated" and a["model"] == "llm-gia" and "[" in a["answer"]
    system, user = llm.prompts[0]
    assert "CHỈ dùng" in system and "[1]" in user and Q in user  # prompt mang đoạn trích đánh số

def test_generate_false_never_calls_llm():
    rag, llm = rag_with("bất kỳ [1].")
    assert rag.answer(Q, generate=False)["mode"] == "extractive" and llm.prompts == []

def test_off_topic_never_reaches_llm():
    rag, llm = rag_with("Giá vàng hôm nay tăng [1].")
    a = rag.answer("giá vàng hôm nay")
    assert a["evidence"] == [] and llm.prompts == [] and a["mode"] == "extractive"

@pytest.mark.parametrize("reply, why", [
    ("Hãy ăn 15 gam đường và nghỉ ngơi.", "không có trích dẫn"),
    ("Hãy ăn carbohydrate tác dụng nhanh [7].", "không tồn tại"),
    ("Ăn carbohydrate tác dụng nhanh [1]. Sau đó tiêm thêm insulin để cân bằng.", "câu không có trích dẫn"),
    ("Hãy tiêm 8 đơn vị insulin ngay [1].", "số liệu"),
    ("Cơ sở tri thức không có thông tin này.", "không đủ thông tin"),
    ("Nên uống trà gừng mật ong ấm trước khi ngủ để thư giãn [1].", "bám đoạn trích"),
    ("", "rỗng"),
])
def test_unsupported_answers_fall_back_to_excerpts(reply, why):
    rag, _ = rag_with(reply)
    a = rag.answer(Q)
    assert a["mode"] == "extractive" and why in a["fallback_reason"]
    assert a["answer"] == rag.extractive(rag.retrieve(Q))  # người dùng chỉ thấy đoạn trích gốc
    assert a["guard"]["raw"] == reply  # vẫn lưu câu bị chặn để giải thích khi demo

def test_llm_error_falls_back():
    rag, _ = rag_with(OSError("connection refused"))
    a = rag.answer(Q)
    assert a["mode"] == "extractive" and "không gọi được LLM" in a["fallback_reason"]

def test_reasoning_tags_removed():
    rag, llm = rag_with("")
    llm.reply = "<think>suy nghĩ nội bộ</think>" + grounded_reply(rag)
    a = rag.answer(Q)
    assert a["mode"] == "generated" and "think" not in a["answer"]

def test_verify_accepts_citation_after_period_and_decimal_comma():
    ev = RAG(KB).retrieve("tiêu chuẩn chẩn đoán tiền đái tháo đường HbA1c")
    i = next(n for n, e in enumerate(ev) if "tiền đái tháo đường" in e.title.lower()) + 1
    ok, reason, _ = verify(f"HbA1c từ 5,7 đến 6,4 phần trăm được chẩn đoán tiền tiểu đường. [{i}]", ev)
    assert ok, reason

def test_prompt_numbers_every_excerpt():
    ev = RAG(KB).retrieve(Q)
    p = build_prompt(Q, ev)
    assert all(f"[{i + 1}] {e.title}" in p for i, e in enumerate(ev))

def test_generator_from_env(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert generator_from_env() is None
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    assert generator_from_env().name == "qwen2.5:7b"


# ---- truy xuất vector ----

def test_lsa_vectors_are_unit_length():
    idx = VectorIndex(KB.recommendations, LSAEmbedder())
    import numpy as np
    assert np.allclose(np.linalg.norm(idx.matrix, axis=1), 1, atol=1e-5)
    assert idx.matrix.shape[0] == len(KB.recommendations)

@pytest.mark.parametrize("method", ["vector", "hybrid"])
def test_vector_and_hybrid_find_hypo_treatment(method):
    titles = [e.title for e in RAG(KB, retriever=method).retrieve(Q)]
    assert any("hạ đường huyết" in t.lower() for t in titles)

@pytest.mark.parametrize("method", ["vector", "hybrid"])
def test_vector_respects_off_topic_filter(method):
    assert RAG(KB, retriever=method).retrieve("giá vàng hôm nay") == []

def test_unknown_retriever_rejected():
    with pytest.raises(ValueError, match="RAG_RETRIEVER"):
        RAG(KB, retriever="faiss")

def test_rrf_rewards_agreement():
    assert rrf([1, 2, 3], [3, 1, 2])[0][1] == 1

def test_broken_embedding_endpoint_falls_back_to_lsa():
    class Dead:
        name = "embedding chết"
        def fit(self, texts): return self
        def embed(self, texts): raise OSError("không kết nối được")
    assert VectorIndex(KB.recommendations, Dead()).embedder.name.startswith("LSA")

def test_eval_reports_all_retrievers():
    from diabetes_mas.eval.rag_eval import evaluate
    r = evaluate()
    assert {"word_overlap", "bm25", "vector", "hybrid"} <= set(r)
    assert r["vector"]["recall_at_3"] > r["word_overlap"]["recall_at_3"]
