"""LLM chỉ được diễn đạt lại: mọi con số phải có trong nguồn; số kèm đơn vị phải khớp cả đơn vị."""
import re

UNITS = {"đơn vị": "dv", "đv": "dv", "iu": "dv", "units": "dv", "unit": "dv", "u": "dv", "viên": "vien",
         "mg/dl": "mg/dl", "mmol/l": "mmol/l", "mg": "mg", "ml": "ml", "kcal": "kcal", "phút": "phut", "giờ": "gio",
         "g": "g", "%": "%"}
UNIT_RE = "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True))
WITH_UNIT = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({UNIT_RE})(?!\w)", re.I)
ANY_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)
SPELLED = re.compile(r"\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(đơn vị|đv|viên)\b", re.I)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _pairs(text: str) -> set[tuple[float, str]]:
    return {(_num(m.group(1)), UNITS[m.group(2).lower()]) for m in WITH_UNIT.finditer(text)}


def unsupported_numbers(llm_text: str, source_text: str) -> list[str]:
    text = LIST_MARKER.sub("", llm_text)
    known_pairs, known = _pairs(source_text), {_num(x) for x in ANY_NUMBER.findall(source_text)}
    bad = [m.group(0) for m in WITH_UNIT.finditer(text) if (_num(m.group(1)), UNITS[m.group(2).lower()]) not in known_pairs]
    bad += [x for x in ANY_NUMBER.findall(text) if _num(x) not in known]
    bad += [m.group(0) for m in SPELLED.finditer(text) if m.group(0).lower() not in source_text.lower()]
    return bad


def is_grounded(llm_text: str, source_text: str) -> tuple[bool, list[str]]:
    bad = unsupported_numbers(llm_text, source_text)
    return not bad, bad
