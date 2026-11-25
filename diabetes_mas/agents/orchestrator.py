"""Orchestrator (điều phối): lập kế hoạch theo tình trạng, gọi agent, gọi Auditor, tổng hợp báo cáo."""
import math

from .. import config as C
from ..schemas import LEVEL_ORDER, AnalysisContext, Report
from .auditor import audit
from .diagnosis_agent import DiagnosisAgent
from .exercise_agent import ExerciseAgent
from .glucose_agent import GlucoseAgent
from .medication_agent import MedicationAgent
from .nutrition_agent import NutritionAgent


def validate_readings(readings, unit_confirmed: bool = False) -> None:
    if not readings:
        raise ValueError("Cần ít nhất 1 số đo đường huyết.")
    if any(not math.isfinite(float(x)) for x in readings):
        raise ValueError("Số đo đường huyết không hợp lệ.")
    if not unit_confirmed and all(float(x) <= C.MMOL_SUSPECT_MAX for x in readings):
        raise ValueError("Giá trị có vẻ là mmol/L, hệ thống cần mg/dL (1 mmol/L ≈ 18 mg/dL): hãy chọn đúng đơn vị. "
                         "Nếu thực sự là mg/dL thì đây là mức nguy hiểm: dùng ngay carbohydrate nhanh và gọi cấp cứu.")
    if any(not C.GLUCOSE_MIN_VALID <= float(x) <= C.GLUCOSE_MAX_VALID for x in readings):
        raise ValueError(f"Số đo phải trong khoảng {C.GLUCOSE_MIN_VALID}-{C.GLUCOSE_MAX_VALID} mg/dL.")


class Orchestrator:
    def __init__(self, predictor, kb, rag, policy):
        self.predictor, self.kb, self.rag = predictor, kb, rag
        self.agents = {"diagnosis": DiagnosisAgent(kb, rag), "glucose": GlucoseAgent(kb, rag), "nutrition": NutritionAgent(kb, rag),
                       "medication": MedicationAgent(kb, rag, policy), "exercise": ExerciseAgent(kb, rag)}

    def plan(self, ctx: AnalysisContext) -> tuple[str, list[str]]:
        has_meds = bool(ctx.profile.medications or ctx.profile.on_insulin)
        if ctx.glucose < C.HYPO_L2 or ctx.glucose >= C.HYPER_CRITICAL:
            ctx.shared["review_only"] = True
            kind = "hạ đường huyết nặng" if ctx.glucose < C.HYPO_L2 else "tăng đường huyết rất cao"
            # vẫn rà soát thuốc: sulfonylurea trên nền suy thận là nguyên nhân hạ đường huyết hay gặp
            return f"KHẨN CẤP ({kind})", ["diagnosis", "glucose"] + (["medication"] if has_meds else [])
        steps = ["diagnosis", "glucose", "nutrition"]
        if has_meds:
            steps.append("medication")
        return "Thường quy", steps + ["exercise"]

    def analyze(self, profile, readings, planned_carbs=None, minutes_since_insulin=None, last_insulin_units=None,
                reading_age_min=None, meal_type=None, unit_confirmed=False) -> Report:
        validate_readings(readings, unit_confirmed)
        readings = [float(x) for x in readings]
        ctx = AnalysisContext(profile, readings, self.predictor.classify(profile), self.predictor.forecast(readings),
                              planned_carbs, minutes_since_insulin, last_insulin_units, reading_age_min, meal_type)
        plan, steps = self.plan(ctx)
        results = {s: self.agents[s].run(ctx) for s in steps}
        for s, r in results.items():
            r.kind = self.agents[s].kind
        findings = audit(ctx, results)

        seen, alerts = set(), []
        for r in results.values():
            for a in r.alerts:
                if a.message not in seen:
                    seen.add(a.message)
                    alerts.append(a)
        alerts.sort(key=lambda a: LEVEL_ORDER[a.level])
        sources = sorted({f"{e.source} ({e.url})" for r in results.values() for e in r.evidence})
        return Report(plan, steps, ctx.classification, ctx.forecast, results, alerts,
                      [f"{f.rule}: {f.message}" for f in findings], sources, C.DISCLAIMER, readings)
