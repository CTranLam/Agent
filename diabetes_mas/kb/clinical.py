"""Phần lâm sàng của Knowledge Graph: Disease, Symptom, RiskFactor, Measurement, Patient.

Tri thức (bệnh, triệu chứng, yếu tố nguy cơ, ngưỡng chỉ số) nằm trong clinical.json.
Bệnh nhân là các ca mẫu giả lập trong scenarios.py, được chuyển thành nút Patient nối tới
chỉ số đo, yếu tố nguy cơ, bệnh lý nền (Condition) và thuốc đang dùng (Drug) đã có trong KB.
"""
import json
import re
from functools import lru_cache

from ..agents.conditions import evaluate
from ..schemas import PatientProfile
from .store import HERE, KnowledgeBase

RISK_RULES = {
    "overweight": lambda p, drugs: p.bmi >= 25,
    "hypertension": lambda p, drugs: bool(p.hypertension),
    "heart_disease": lambda p, drugs: bool(p.heart_disease or p.heart_failure),
    "smoking": lambda p, drugs: p.smoking in ("current", "former", "ever", "not current"),
    "age_ge_45": lambda p, drugs: p.age >= 45,
    "long_duration": lambda p, drugs: p.duration_years >= 10,
    "poor_control": lambda p, drugs: p.hba1c >= 8,
    "hypo_drug": lambda p, drugs: p.on_insulin or any(d.startswith("insulin") or d in ("gliclazide",) for d in drugs),
    # alcohol_fasting, strict_diet: hồ sơ chưa có trường này, chỉ là tri thức trong đồ thị
}


@lru_cache(maxsize=1)
def load_clinical() -> dict:
    return json.loads((HERE / "clinical.json").read_text(encoding="utf-8"))


def patient_record(name: str, title: str, p: PatientProfile, readings: list[float], kb: KnowledgeBase) -> dict:
    entries = [x.strip() for raw in p.medications for x in re.split(r"[;,+/]|\s+và\s+", raw, flags=re.I)]
    drugs = []
    for e in filter(None, entries):
        d = kb.match_drug(e)
        if d and d["name"] not in drugs:
            drugs.append(d["name"])
    codes = sorted({c for d in kb.drugs for c in d["contraindications"] + d["cautions"]})
    dtype = p.diabetes_type_known
    measurements = [("hba1c", p.hba1c), ("fpg", p.fasting_glucose), ("glucose", readings[-1] if readings else None),
                    ("egfr", p.egfr), ("bmi", round(p.bmi, 1)), ("c_peptide", p.c_peptide)]
    return {"name": name, "title": title, "age": p.age, "sex": p.sex, "diagnosis": dtype,
            "measurements": [{"code": c, "value": float(v)} for c, v in measurements if v is not None],
            "risk_factors": [c for c, rule in RISK_RULES.items() if rule(p, drugs)],
            "conditions": [c for c in codes if evaluate(c, p, dtype)],
            "drugs": drugs}


def demo_patients(kb: KnowledgeBase) -> list[dict]:
    from ..scenarios import SCENARIOS
    return [patient_record(s["profile"].name or f"Ca {k}", s["title"], s["profile"], s["readings"], kb) for k, s in SCENARIOS.items()]
