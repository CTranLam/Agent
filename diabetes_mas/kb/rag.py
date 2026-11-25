"""Truy xuất lai: BM25 (từ đơn + bigram) + mở rộng đồng nghĩa + khớp chủ đề."""
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

from ..schemas import Evidence
from .generate import Generator, generator_from_env
from .store import CYPHER_FILE, CYPHER_HEADER, KnowledgeBase, fold
from .vector import VectorIndex, embedder_from_env, rrf

log = logging.getLogger(__name__)

K1, B = 1.5, 0.75
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
QUESTION_SYNONYMS = {"lam gi": "xu tri", "the nao": "xu tri", "lam sao": "xu tri"}
TOPIC_KEYWORDS = {
    "hypoglycemia": ["ha duong huyet"], "hyperglycemia": ["tang duong huyet", "ceton"],
    "nutrition": ["dinh duong"], "exercise": ["van dong"],
    "medication": ["thuoc", "metformin", "gliclazide", "pioglitazone", "liraglutide"],
    "insulin": ["insulin", "tiem"], "complications": ["bien chung", "ban chan", "than", "mat"],
    "special_groups": ["mang thai", "cao tuoi", "tre em"],
}
STOP = set("""bi la va cua co khong thi ma o toi minh ban nay hom gi nao the phai lam nhu sao nen duoc cho
            mot cac nhung voi khi neu de tu ve trong ngoai tren duoi rat nhieu it ai dau""".split())
# Âm tiết phổ biến ngoài y khoa ("đường" cao tốc, "chứng" khoán, tiền "lương").
GENERIC = {"duong", "huyet", "chung", "luong", "tien", "khoan"}
OFF_TOPIC = ("cao toc",)  # shortcut: danh sách tay, chưa lọc hết (vd "nhập khẩu thuốc lá"); cần chính xác thì dùng embedding
TREATMENT_TITLE = "Xử trí hạ đường huyết theo quy tắc 15-15"
DOMAIN_SHORT = {"carb", "lieu", "muoi", "man", "thuoc", "tiem", "tap"}
MIN_UNIGRAM_LEN = 5  # shortcut: lọc ngoài phạm vi bằng độ dài từ; nếu cần chính xác hơn thì dùng embedding
RETRIEVERS = {"bm25": "BM25 + bigram + đồng nghĩa", "vector": "vector (embedding, cosine)", "hybrid": "lai BM25 + vector (RRF)"}


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


_PROP_KEY = re.compile(r"\s*,?\s*`(\w+)`:\s*")
_NODE = re.compile(r"^CREATE \(:`(\w+)` \{")
_EDGE = re.compile(r"^MATCH \(a \{_kid: (\d+)\}\), \(b \{_kid: (\d+)\}\) CREATE \(a\)-\[:`(FROM|ABOUT)` ")


def _node_props(line: str, start: int) -> dict:
    """Thuộc tính do export_kb ghi: tên trong dấu huyền, giá trị dạng JSON. Không dùng eval."""
    props, decoder, i = {}, json.JSONDecoder(), start
    while line[i] != "}":
        key = _PROP_KEY.match(line, i)
        if not key:
            raise ValueError(f"Dòng .cypher sai định dạng: {line[:80]}")
        props[key.group(1)], i = decoder.raw_decode(line, key.end())
    return props


def read_cypher_recommendations(path) -> list[dict]:
    """Đọc khuyến nghị thẳng từ file export_kb đã xuất"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != CYPHER_HEADER:
        raise ValueError(f"{path} không phải file KB do export_kb xuất ra.")
    nodes, links = {}, {}
    for line in lines[1:]:
        if m := _NODE.match(line):
            if m.group(1) in ("Recommendation", "Source", "Topic"):
                props = _node_props(line, m.end())
                nodes[props["_kid"]] = props
        elif m := _EDGE.match(line):
            links[(int(m.group(1)), m.group(3))] = int(m.group(2))
    recs = sorted((n for n in nodes.values() if "title" in n), key=lambda n: n["seq"])
    if not recs:
        raise ValueError(f"{path} không có khuyến nghị nào.")
    out = []
    for n in recs:
        source, topic = links.get((n["_kid"], "FROM")), links.get((n["_kid"], "ABOUT"))
        if source not in nodes or topic not in nodes:
            raise ValueError(f"Khuyến nghị '{n['title']}' thiếu nguồn hoặc chủ đề trong {path}.")
        out.append({"title": n["title"], "text": n["text"], "source": nodes[source]["name"],
                    "url": nodes[source]["url"], "topic": nodes[topic]["name"]})
    return out


def build_rag(kb: KnowledgeBase, path=None) -> "RAG":
    """Chat dùng file .cypher nếu có; chưa xuất file thì dùng KB. File hỏng thì báo lỗi, không âm thầm bỏ qua.
    RAG_RETRIEVER=bm25|vector|hybrid chọn bộ truy xuất; LLM_BASE_URL + LLM_MODEL bật bước sinh câu trả lời."""
    path = Path(path or os.getenv("KB_CYPHER", CYPHER_FILE))
    opts = {"retriever": os.getenv("RAG_RETRIEVER", "bm25"), "embedder": embedder_from_env(), "generator": generator_from_env()}
    if not path.exists():
        log.warning("Không có %s, RAG dùng dữ liệu KB.", path)
        return RAG(kb, **opts)
    return RAG(kb, read_cypher_recommendations(path), "file .cypher", **opts)


class RAG:
    def __init__(self, kb: KnowledgeBase, docs: list[dict] | None = None, origin: str = "KB",
                 retriever: str = "bm25", embedder=None, generator: Generator | None = None):
        if retriever not in RETRIEVERS:
            raise ValueError(f"RAG_RETRIEVER không hợp lệ: {retriever} (chọn {', '.join(RETRIEVERS)})")
        self.kb, self.origin, self.retriever, self.generator = kb, origin, retriever, generator
        self._embedder, self._index = embedder, None
        self.docs = kb.recommendations if docs is None else docs
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

    def in_scope(self, query: str) -> bool:
        """Bộ lọc ngoài phạm vi dùng chung cho mọi bộ truy xuất (BM25, vector, lai)."""
        if any(p in " ".join(tokens(query)) for p in OFF_TOPIC):
            return False
        # ngoài phạm vi: phải khớp bigram hoặc một từ đủ dài, không chỉ vài âm tiết rời
        unexpanded = set(tokens(query))
        useful = {t for t in _query_features(query, question_words=False)
                  if t not in GENERIC and (t in self._idf or t in unexpanded & DOMAIN_SHORT) and ("_" in t or len(t) >= MIN_UNIGRAM_LEN or t in DOMAIN_SHORT)}
        return bool(useful)

    @property
    def index(self) -> VectorIndex:
        """Kho vector dựng lần đầu khi cần, để chế độ BM25 không tốn thời gian khởi động."""
        if self._index is None:
            self._index = VectorIndex(self.docs, self._embedder)
        return self._index

    def _evidence(self, i: int, score: float) -> Evidence:
        d = self.docs[i]
        return Evidence(d["source"], d["title"], d["text"], round(score, 3), d["url"])

    def retrieve(self, query: str, k: int = 3, topic: str | None = None, method: str | None = None) -> list[Evidence]:
        method = method or self.retriever
        if not self.in_scope(query):
            return []
        if method == "bm25":
            hits = self._bm25_ranked(query, topic)
        elif method == "vector":
            hits = self.index.search(expand_query(query))
        else:
            hits = rrf([i for _, i in self._bm25_ranked(query, topic)], [i for _, i in self.index.search(expand_query(query))])
        return [self._evidence(i, s) for s, i in hits[:k]]

    def _bm25_ranked(self, query: str, topic: str | None = None) -> list[tuple[float, int]]:
        q = Counter(_query_features(query))
        raw = [self._bm25(q, i) for i in range(len(self.docs))]
        top = max(raw)
        if top <= 0:
            return []
        expanded = " ".join(tokens(expand_query(query)))
        hits = {t for t, kws in TOPIC_KEYWORDS.items() if any(w in expanded for w in kws)}
        if topic:
            hits.add(topic)
        # câu nào về hạ đường huyết cũng phải thấy được đoạn xử trí
        scored = [(raw[i] / top + (0.5 if self.docs[i]["topic"] in hits else 0.0)
                   + (0.6 if "hypoglycemia" in hits and self.docs[i]["title"] == TREATMENT_TITLE else 0.0), i)
                  for i in range(len(self.docs)) if raw[i] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    def extractive(self, evidence: list[Evidence]) -> str:
        if not evidence:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức. Hãy hỏi về tiểu đường hoặc hỏi bác sĩ."
        return "\n\n".join(f"{i + 1}. {e.title}\n{e.text}\n(Nguồn: {e.source})" for i, e in enumerate(evidence))

    def answer(self, question: str, generate: bool | None = None) -> dict:
        """generate=None: theo cấu hình server; False: chỉ trích; True: sinh nếu đã cấu hình LLM."""
        ev = self.retrieve(question)
        out = {"answer": self.extractive(ev), "evidence": [vars(e) for e in ev], "mode": "extractive",
               "retriever": RETRIEVERS[self.retriever]}
        if generate is False or not ev:
            return out  # không có đoạn trích thì không gọi LLM: tránh trả lời ngoài cơ sở tri thức
        if self.generator is None:
            if generate:
                out["fallback_reason"] = "chưa cấu hình LLM (LLM_BASE_URL, LLM_MODEL)"
            return out
        g = self.generator.generate(question, ev)
        out["guard"] = g.guard
        if g.ok:
            out.update(answer=g.text, mode="generated", model=self.generator.name)
        else:
            out["fallback_reason"] = g.reason
        return out
