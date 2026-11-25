"""PDF -> KB: trích các đoạn khai trong knowledge_docs/pdf/sources.json thành knowledge_docs/pdf_*.md.

python -m diabetes_mas.kb.pdf_docs      chạy lại khi đổi PDF hoặc sources.json, rồi nạp và xuất lại KB
"""
import json
import re

from pypdf import PdfReader

from .store import DOCS

PDF_DIR = DOCS / "pdf"
# Lớp chữ của PDF làm rơi chữ "ỡ" và số mũ (2-8°C thành 2-80C): đoạn nào còn dấu hiệu này thì không được ghi vào KB.
BROKEN = re.compile(r"\d0\s?C\b|\b(?:dư|ngư) ng\b|\bm (?=[,.]|$)|[\x00-\x08\x0b-\x1f-�]")
MIN_CHARS = 80


def clean(text: str, fixes: dict[str, str]) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    for wrong, right in fixes.items():
        text = text.replace(wrong, right)
    if m := BROKEN.search(text):
        raise ValueError(f"Chữ trích từ PDF bị lỗi gần '{text[max(0, m.start() - 30):m.end() + 10]}': thêm vào 'fixes'.")
    if len(text) < MIN_CHARS:  # bảng hay bị PDF đẩy xuống cuối trang, chỉ còn lại tiêu đề
        raise ValueError(f"Đoạn trích quá ngắn, kiểm tra lại mốc: '{text}'")
    return text[0].upper() + text[1:]


def cut(page_text: str, start: str, end: str | None) -> str:
    """Đoạn từ mốc start đến trước mốc end (None: hết trang). Mốc thiếu hoặc lặp thì báo lỗi, không đoán."""
    t = re.sub(r"\s+", " ", page_text)
    if t.count(start) != 1:
        raise ValueError(f"Mốc đầu '{start}' xuất hiện {t.count(start)} lần, cần đúng 1.")
    t = t[t.index(start):]
    if end is None:
        return t
    if end not in t:
        raise ValueError(f"Không thấy mốc cuối '{end}' sau '{start}'.")
    return t[:t.index(end)]


def build(src: dict) -> str:
    pages = PdfReader(PDF_DIR / src["file"]).pages
    blocks = []
    for s in src["sections"]:
        n = s["page"]
        text = clean(cut(pages[n - 1].extract_text(), s["start"], s.get("end")), src["fixes"])
        # trang in trên giấy lệch với trang của file vì có bìa; link dùng trang của file
        blocks.append(f"## {s['title']}\nsource: {src['source']}, trang {n - src['page_offset']}\n"
                      f"url: {src['url']}#page={n}\ntopic: {s['topic']}\n{text}\n")
    return "\n".join(blocks)


def sources() -> list[dict]:
    return json.loads((PDF_DIR / "sources.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    for src in sources():
        md = build(src)
        (DOCS / src["out"]).write_text(md, encoding="utf-8")
        print(f"{src['file']}: đã trích {len(src['sections'])} đoạn vào {DOCS / src['out']}")
