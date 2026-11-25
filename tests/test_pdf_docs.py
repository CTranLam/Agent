"""Nguồn PDF: file .md đã ghi phải đúng bằng nội dung trích lại từ PDF, và lỗi lớp chữ không được lọt vào KB."""
import pytest
from diabetes_mas.kb.pdf_docs import build, clean, cut, sources
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import DOCS, load_kb


@pytest.mark.parametrize("src", sources(), ids=lambda s: s["file"])
def test_md_matches_pdf(src):
    assert (DOCS / src["out"]).read_text(encoding="utf-8") == build(src)

@pytest.mark.parametrize("bad", ["để ở 2-80C", "nhiệt độ > 300C", "chế độ dinh dư ng", "thịt m , nội tạng", "\uf0b7 Đo 4 lần"])
def test_broken_text_layer_is_rejected(bad):
    with pytest.raises(ValueError, match="bị lỗi"):
        clean(bad + " " + "x" * 100, {})

def test_heading_without_content_is_rejected():
    with pytest.raises(ValueError, match="quá ngắn"):
        clean("Bảng 9: Các mục tiêu", {})

def test_markers_must_be_exact():
    assert cut("a  Đầu x\ny Cuối z", "Đầu", "Cuối") == "Đầu x y "
    for start, end in (("Không có", None), ("Đầu", "Không có")):
        with pytest.raises(ValueError):
            cut("Đầu x Cuối", start, end)
    with pytest.raises(ValueError, match="2 lần"):
        cut("Đầu a Đầu b", "Đầu", None)

def test_insulin_storage_temperature_is_safe():
    text = next(r["text"] for r in load_kb().recommendations if r["title"] == "Bảo quản insulin")
    assert "2-8°C" in text and "<30°C" in text

@pytest.mark.parametrize("q,title", [("bảo quản insulin trong tủ lạnh thế nào", "Bảo quản insulin"),
                                     ("mục tiêu đường huyết trong thai kỳ", "Mục tiêu đường huyết trong thai kỳ"),
                                     ("bao lâu phải khám mắt một lần", "Lịch khám mắt")])
def test_rag_answers_from_pdf(q, title):
    ev = RAG(load_kb()).retrieve(q)
    assert title in [e.title for e in ev] and any("#page=" in e.url for e in ev)
