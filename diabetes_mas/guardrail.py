"""LLM chỉ được diễn đạt lại: mọi số đi kèm đơn vị phải có trong dữ liệu nguồn."""
import re

UNIT = r"(?:đơn vị|đv|mg/dl|mmol/l|mg|ml|kcal|phút|giờ|g|%)"
WITH_UNIT = re.compile(rf"(\d+(?:[.,]\d+)?)\s*{UNIT}(?!\w)", re.I)
ANY_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def unsupported_numbers(llm_text: str, source_text: str) -> list[str]:
    # shortcut: số không kèm đơn vị không bị kiểm; nếu cần chặt hơn thì kiểm mọi số
    known = {_num(x) for x in ANY_NUMBER.findall(source_text)}
    return [m.group(0) for m in WITH_UNIT.finditer(llm_text) if _num(m.group(1)) not in known]


def is_grounded(llm_text: str, source_text: str) -> tuple[bool, list[str]]:
    bad = unsupported_numbers(llm_text, source_text)
    return not bad, bad
