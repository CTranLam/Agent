"""Truy xuất lai: BM25 (từ đơn + bigram) + mở rộng đồng nghĩa + khớp chủ đề."""
import math
import re
from collections import Counter

from ..guardrail import is_grounded
from ..schemas import Evidence
from .store import KnowledgeBase, fold

K1, B = 1.5, 0.75
ANSWER_SYSTEM = ("Bạn là trợ lý tiểu đường. Chỉ trả lời bằng tiếng Việt, chỉ dựa vào phần Bằng chứng; không thêm con số, "
                 "liều thuốc hay tên thuốc ngoài bằng chứng. Nếu bằng chứng không đủ thì nói rõ và khuyên hỏi bác sĩ.")
# Cách nói đời thường -> thuật ngữ trong KB. Khoá và giá trị đều đã bỏ dấu.
SYNONYMS = {
    "tut duong": "ha duong huyet hypoglycemia", "tut duong huyet": "ha duong huyet hypoglycemia",
    "duong huyet thap": "ha duong huyet", "duong thap": "ha duong huyet",
    "duong huyet cao": "tang duong huyet", "duong cao": "tang duong huyet",
    "tieu duong": "dai thao duong", "an gi": "dinh duong tinh bot", "an uong": "dinh duong tinh bot",
    "che do an": "dinh duong", "tap the duc": "van dong", "tap luyen": "van dong", "the thao": "van dong",
    "chich": "tiem", "bau": "mang thai", "co thai": "mang thai", "nguoi gia": "nguoi cao tuoi",
    "gan chan": "ban chan", "dau chan": "ban chan", "suy than": "than egfr", "ruou": "ruou bia",
    "man": "muoi", "keto": "ceton",
}
# Từ để hỏi: giúp xếp hạng, nhưng không chứng tỏ câu hỏi thuộc phạm vi tiểu đường.
QUESTION_SYNONYMS = {"lam gi": "xu tri", "the nao": "xu tri"}
TOPIC_KEYWORDS = {
    "hypoglycemia": ["ha duong huyet"], "hyperglycemia": ["tang duong huyet", "ceton"],
    "nutrition": ["dinh duong"], "exercise": ["van dong"],
    "medication": ["thuoc", "metformin", "gliclazide", "pioglitazone", "liraglutide"],
    "insulin": ["insulin", "tiem"], "complications": ["bien chung", "ban chan", "than", "mat"],
    "special_groups": ["mang thai", "cao tuoi", "tre em"],
}
STOP = set("""bi la va cua co khong thi ma o toi minh ban nay hom gi nao the phai lam nhu sao nen duoc cho
            mot cac nhung voi khi neu de tu ve trong ngoai tren duoi rat nhieu it ai dau""".split())
DOMAIN_SHORT = {"carb", "lieu", "muoi", "man", "thuoc", "tiem"}
MIN_UNIGRAM_LEN = 5  # shortcut: lọc ngoài phạm vi bằng độ dài từ; nếu cần chính xác hơn thì dùng embedding


def tokens(s: str) -> list[str]:
    return re.findall(r"\w+", fold(s))


def expand_query(q: str, question_words: bool = True) -> str:
    f = " " + " ".join(tokens(q)) + " "
    table = {**SYNONYMS, **QUESTION_SYNONYMS} if question_words else SYNONYMS
    extra = [add for phrase, add in table.items() if f" {phrase} " in f]
    return q if not extra else q + " " + " ".join(extra)


def features(s: str) -> list[str]:
    t = tokens(s)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


def _query_features(q: str, question_words: bool = True) -> list[str]:
    t = tokens(expand_query(q, question_words))
    return [x for x in t if x not in STOP] + [f"{a}_{b}" for a, b in zip(t, t[1:])]


class RAG:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.docs = kb.recommendations
        self._tf = [Counter(features(d["title"]) * 2 + features(d["text"])) for d in self.docs]
        self._len = [sum(c.values()) for c in self._tf]
        self._avg = sum(self._len) / len(self._len)
        df = Counter(t for c in self._tf for t in c)
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    def _bm25(self, q: Counter, i: int) -> float:
        tf, s = self._tf[i], 0.0
        for t in q:
            f = tf.get(t)
            if f:
                s += self._idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self._len[i] / self._avg))
        return s

    def retrieve(self, query: str, k: int = 3, topic: str | None = None) -> list[Evidence]:
        q = Counter(_query_features(query))
        # ngoài phạm vi: phải khớp bigram hoặc một từ đủ dài, không chỉ vài âm tiết rời
        unexpanded = set(tokens(query))
        useful = {t for t in _query_features(query, question_words=False)
                  if (t in self._idf or t in unexpanded & DOMAIN_SHORT) and ("_" in t or len(t) >= MIN_UNIGRAM_LEN or t in DOMAIN_SHORT)}
        if not useful:
            return []
        raw = [self._bm25(q, i) for i in range(len(self.docs))]
        top = max(raw)
        if top <= 0:
            return []
        expanded = " ".join(tokens(expand_query(query)))
        hits = {t for t, kws in TOPIC_KEYWORDS.items() if any(w in expanded for w in kws)}
        if topic:
            hits.add(topic)
        scored = [(raw[i] / top + (0.5 if self.docs[i]["topic"] in hits else 0.0), i)
                  for i in range(len(self.docs)) if raw[i] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Evidence(self.docs[i]["source"], self.docs[i]["title"], self.docs[i]["text"], round(s, 3))
                for s, i in scored[:k]]

    def extractive(self, evidence: list[Evidence]) -> str:
        if not evidence:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức. Hãy hỏi về tiểu đường hoặc hỏi bác sĩ."
        return "\n\n".join(f"{i + 1}. {e.title}\n{e.text}\n(Nguồn: {e.source})" for i, e in enumerate(evidence))

    def answer(self, question: str, llm=None) -> dict:
        ev = self.retrieve(question)
        result = {"answer": self.extractive(ev), "evidence": [vars(e) for e in ev], "mode": "extractive", "note": None}
        if not ev or llm is None or not llm.enabled:
            return result
        src = "\n".join(f"[{e.source}] {e.title}: {e.text}" for e in ev)
        out = llm.complete(ANSWER_SYSTEM, f"Bằng chứng:\n{src}\n\nCâu hỏi: {question}")
        if out is None:
            result["note"] = f"LLM lỗi ({llm.last_error}), dùng trích nguyên văn."
        elif is_grounded(out, src)[0]:
            result.update(answer=out, mode="llm")
        else:
            result["note"] = "Đã bỏ câu trả lời của LLM vì chứa số liệu ngoài bằng chứng."
        return result
