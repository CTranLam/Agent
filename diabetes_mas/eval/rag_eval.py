"""So đếm từ trùng, BM25, vector (embedding) và lai: python -m diabetes_mas.eval.rag_eval"""
import json
from pathlib import Path

from ..kb.rag import RAG, tokens
from ..kb.store import load_kb
from ..kb.vector import embedder_from_env

GOLD = Path(__file__).parent / "rag_gold.json"


def _overlap_rank(docs, q) -> list[str]:
    qs = set(tokens(q))
    scored = sorted(docs, key=lambda d: -len(qs & set(tokens(d["title"] + " " + d["text"]))))
    return [d["title"] for d in scored]


def _score(ranker, gold) -> dict:
    hit, rr = 0, 0.0
    for g in gold:
        ranks = [i for i, t in enumerate(ranker(g["q"])) if t in g["gold"]]
        if ranks:
            hit += ranks[0] < 3
            rr += 1 / (ranks[0] + 1)
    return {"recall_at_3": round(hit / len(gold), 3), "mrr": round(rr / len(gold), 3)}


def evaluate() -> dict:
    kb, gold = load_kb(), json.loads(GOLD.read_text(encoding="utf-8"))
    rag = RAG(kb, embedder=embedder_from_env())
    n = len(kb.recommendations)
    ranker = lambda m: (lambda q: [e.title for e in rag.retrieve(q, k=n, method=m)])
    return {"n_questions": len(gold),
            "word_overlap": _score(lambda q: _overlap_rank(kb.recommendations, q), gold),
            "bm25": _score(ranker("bm25"), gold),
            "vector": {**_score(ranker("vector"), gold), "embedder": rag.index.embedder.name},
            "hybrid": _score(ranker("hybrid"), gold),
            "caveat": "Bộ câu hỏi do chính người viết KB soạn nên số đo có xu hướng lạc quan; "
                      "bộ lọc từ chuyên môn ngắn được thêm sau khi thấy câu trượt."}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
