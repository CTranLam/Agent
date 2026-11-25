"""Bước "G" của RAG: sinh câu trả lời bằng LLM, CHỈ dựa trên đoạn trích đã truy xuất, kèm trích dẫn [n].

Tuỳ chọn, tắt mặc định. Bật bằng biến môi trường (endpoint kiểu OpenAI /chat/completions):
    LLM_BASE_URL=http://127.0.0.1:11434/v1   LLM_MODEL=qwen2.5:7b      (Ollama chạy trên máy, dữ liệu không rời máy)
    LLM_BASE_URL=https://api.openai.com/v1   LLM_MODEL=gpt-4o-mini  LLM_API_KEY=...

Chống ảo giác (hallucination) theo ba lớp:
1. Prompt: chỉ dùng đoạn trích, mỗi câu phải có [n], không đủ thông tin thì nói không có.
2. Đầu vào: chỉ gọi LLM khi bộ truy xuất tìm được đoạn trích (câu ngoài phạm vi bị chặn trước).
3. Kiểm tra sau khi sinh (verify): mọi câu phải có trích dẫn hợp lệ, mọi con số phải có trong đoạn được trích,
   đa số từ nội dung của câu phải xuất hiện trong đoạn được trích. Trượt bất kỳ điều nào thì KHÔNG hiển thị
   câu trả lời sinh ra mà quay về trả nguyên đoạn trích (extractive).
"""
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..schemas import Evidence
from .store import fold

SYSTEM = """Bạn là trợ lý giáo dục về bệnh tiểu đường trong một đồ án sinh viên, không phải bác sĩ.
Quy tắc bắt buộc:
1. CHỈ dùng thông tin trong các đoạn trích được đánh số [1], [2], ... bên dưới. Không dùng kiến thức bên ngoài.
2. Mỗi câu trả lời phải kết thúc bằng trích dẫn dạng [n] chỉ tới đoạn trích chứa thông tin của câu đó.
3. Nếu các đoạn trích không đủ để trả lời, chỉ viết đúng một câu: "Cơ sở tri thức không có thông tin này."
4. Không tự đưa ra liều thuốc, liều insulin hay con số nào không có trong đoạn trích.
5. Trả lời bằng tiếng Việt, ngắn gọn, tối đa 5 câu, không dùng tiêu đề hay gạch đầu dòng."""
REFUSAL = "co so tri thuc khong co thong tin"
MIN_SUPPORT = 0.5  # tỉ lệ từ nội dung của mỗi câu phải xuất hiện trong đoạn được trích
_CITE = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_THINK = re.compile(r"<think>.*?</think>", re.S)
# từ nối/từ chức năng không tính khi đo độ bám đoạn trích
_FUNCTION = set("""va la cua co khong thi ma o cho mot cac nhung voi khi neu de tu ve trong ngoai tren duoi rat nhieu it
                   nay do duoc bi can nen phai hay hoac cung da dang se van con theo boi vi nhu sau truoc""".split())


@dataclass
class Generation:
    ok: bool
    text: str = ""
    reason: str = ""
    guard: dict = field(default_factory=dict)


def build_prompt(question: str, evidence: list[Evidence]) -> str:
    blocks = "\n\n".join(f"[{i + 1}] {e.title}\n{e.text}" for i, e in enumerate(evidence))
    return f"Đoạn trích:\n\n{blocks}\n\nCâu hỏi: {question}\n\nTrả lời (mỗi câu kết thúc bằng [n]):"


def _norm_num(s: str) -> str:
    s = s.replace(",", ".")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    out: list[str] = []
    for p in parts:  # "câu. [1]" bị tách thành "câu." và "[1]": gộp lại
        if out and re.fullmatch(r"(\[\d+\][\s,.;]*)+", p):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def _content_words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", fold(_CITE.sub(" ", s))) if len(w) >= 2 and w not in _FUNCTION]


def verify(text: str, evidence: list[Evidence]) -> tuple[bool, str, dict]:
    """Trả (đạt?, lý do nếu trượt, chi tiết kiểm tra)."""
    k = len(evidence)
    sentences = _sentences(text)
    guard = {"n_sentences": len(sentences), "citations": sorted({int(c) for c in _CITE.findall(text)})}
    if not sentences:
        return False, "LLM trả về rỗng", guard
    if REFUSAL in fold(text):
        return False, "LLM báo đoạn trích không đủ thông tin", guard
    if not guard["citations"]:
        return False, "câu trả lời không có trích dẫn", guard
    bad = [c for c in guard["citations"] if not 1 <= c <= k]
    if bad:
        return False, f"trích dẫn đoạn không tồn tại: {bad}", guard
    support = []
    for s in sentences:
        cited = [int(c) for c in _CITE.findall(s)]
        if not cited:
            return False, f"câu không có trích dẫn: \"{s[:60]}\"", guard
        source = " ".join(evidence[c - 1].title + " " + evidence[c - 1].text for c in cited)
        nums_src = {_norm_num(n) for n in _NUM.findall(source)}
        for n in _NUM.findall(_CITE.sub(" ", s)):
            if _norm_num(n) not in nums_src:
                return False, f"số liệu \"{n}\" không có trong đoạn được trích", guard
        words, src_words = _content_words(s), set(_content_words(source))
        support.append(sum(w in src_words for w in words) / len(words) if words else 1.0)
    guard["min_support"] = round(min(support), 2)
    if guard["min_support"] < MIN_SUPPORT:
        return False, f"có câu bám đoạn trích quá ít ({guard['min_support']:.0%} từ có trong nguồn)", guard
    return True, "", guard


class OpenAICompatibleLLM:
    """Gọi POST {base}/chat/completions. Dùng urllib chuẩn để không thêm thư viện."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60):
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "temperature": 0, "max_tokens": 500,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


class Generator:
    def __init__(self, llm, name: str = ""):
        self.llm, self.name = llm, name or getattr(llm, "model", "LLM")

    def generate(self, question: str, evidence: list[Evidence]) -> Generation:
        try:
            raw = self.llm.complete(SYSTEM, build_prompt(question, evidence))
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
            return Generation(False, reason=f"không gọi được LLM ({type(e).__name__})")
        text = _THINK.sub("", raw or "").strip()
        ok, reason, guard = verify(text, evidence)
        return Generation(ok, text if ok else "", reason, {**guard, "raw": text})


def generator_from_env() -> Generator | None:
    base, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
    if not (base and model):
        return None
    return Generator(OpenAICompatibleLLM(base, model, os.getenv("LLM_API_KEY", ""), float(os.getenv("LLM_TIMEOUT", "60"))), model)
