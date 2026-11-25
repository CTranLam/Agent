"""Truy xuất bằng embedding (vector) và truy xuất lai, để so sánh với BM25.

Hai loại embedding:
- LSAEmbedder (mặc định, chạy offline): TF-IDF trên từ đơn + bigram đã bỏ dấu, rồi nén bằng SVD
  (Latent Semantic Analysis) thành vector dày có chuẩn 1. Không cần mạng, không cần tải mô hình.
- HTTPEmbedder (tuỳ chọn): gọi endpoint /embeddings kiểu OpenAI, ví dụ Ollama chạy cục bộ
  (EMBED_BASE_URL=http://127.0.0.1:11434/v1, EMBED_MODEL=nomic-embed-text hoặc bge-m3).

Kho vector là một ma trận numpy trong bộ nhớ (vài chục đoạn nên không cần FAISS hay vector DB);
độ tương đồng là cosine = tích vô hướng của hai vector đã chuẩn hoá.
"""
import json
import logging
import os
import urllib.request

import numpy as np

log = logging.getLogger(__name__)
RRF_K = 60          # hằng số Reciprocal Rank Fusion (Cormack và cộng sự, 2009)
MIN_COSINE = 0.05   # dưới mức này coi như không liên quan


def _doc_text(d: dict) -> str:
    return f"{d['title']}. {d['title']}. {d['text']}"  # tiêu đề nhân đôi như BM25


class LSAEmbedder:
    name = "LSA (TF-IDF + SVD)"

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim, self.seed = dim, seed

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .rag import features
        self.tfidf = TfidfVectorizer(analyzer=features, sublinear_tf=True)
        x = self.tfidf.fit_transform(texts)
        k = max(1, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(x)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.svd.transform(self.tfidf.transform(texts))


class HTTPEmbedder:
    """Endpoint tương thích OpenAI: POST {base}/embeddings {"model", "input": [...]}"""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 30):
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self.name = f"embedding {model}"

    def fit(self, texts: list[str]) -> "HTTPEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        req = urllib.request.Request(f"{self.base_url}/embeddings", method="POST",
                                     data=json.dumps({"model": self.model, "input": texts}).encode(),
                                     headers={"content-type": "application/json",
                                              **({"authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())["data"]
        return np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)


def embedder_from_env():
    base, model = os.getenv("EMBED_BASE_URL"), os.getenv("EMBED_MODEL")
    if base and model:
        return HTTPEmbedder(base, model, os.getenv("EMBED_API_KEY", ""))
    return LSAEmbedder()


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class VectorIndex:
    """Kho vector: mỗi đoạn tri thức (khuyến nghị) là một vector; truy vấn bằng cosine."""

    def __init__(self, docs: list[dict], embedder=None):
        self.docs = docs
        self.embedder = embedder or LSAEmbedder()
        texts = [_doc_text(d) for d in docs]
        try:
            self.matrix = _unit(self.embedder.fit(texts).embed(texts))
        except Exception as e:  # endpoint embedding không chạy: không làm sập server
            if isinstance(self.embedder, LSAEmbedder):
                raise
            log.warning("Không gọi được %s (%s), dùng LSA.", self.embedder.name, e)
            self.embedder = LSAEmbedder()
            self.matrix = _unit(self.embedder.fit(texts).embed(texts))

    def search(self, query: str) -> list[tuple[float, int]]:
        q = _unit(self.embedder.embed([query]))[0]
        sims = self.matrix @ q
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
        return [(float(sims[i]), i) for i in order if sims[i] >= MIN_COSINE]


def rrf(*rankings: list[int], k: int = RRF_K) -> list[tuple[float, int]]:
    """Reciprocal Rank Fusion: cộng 1/(k + hạng) của mỗi đoạn trong từng danh sách."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1 / (k + r + 1)
    return sorted(((s, i) for i, s in score.items()), key=lambda x: (-x[0], x[1]))
