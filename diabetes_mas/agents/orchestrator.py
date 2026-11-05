"""Agent điều phối: lập kế hoạch theo tình trạng, gọi agent, gọi Auditor, tổng hợp báo cáo."""
import json
import math

from .. import config as C
from ..guardrail import is_grounded
from ..schemas import LEVEL_ORDER, AnalysisContext, Report
from .auditor import audit
from .exercise_agent import ExerciseAgent
from .glucose_agent import GlucoseAgent
from .medication_agent import MedicationAgent
from .nutrition_agent import NutritionAgent


SYNTH_SYSTEM = ("Viết bản tư vấn ngắn bằng tiếng Việt cho người bệnh từ dữ liệu JSON của các agent. Chỉ diễn đạt lại, "
                "không thêm con số, liều hay thuốc mới; không bỏ cảnh báo nào.")


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
    def __init__(self, predictor, kb, rag, policy, llm=None):
        self.predictor, self.kb, self.rag, self.llm = predictor, kb, rag, llm
        self.agents = {"glucose": GlucoseAgent(kb, rag), "nutrition": NutritionAgent(kb, rag),
                       "medication": MedicationAgent(kb, rag, policy), "exercise": ExerciseAgent(kb, rag)}

    def plan(self, ctx: AnalysisContext) -> tuple[str, list[str]]:
        has_meds = bool(ctx.profile.medications or ctx.profile.on_insulin)
        if ctx.glucose < C.HYPO_L2 or ctx.glucose >= C.HYPER_CRITICAL:
            ctx.shared["review_only"] = True
            kind = "hạ đường huyết nặng" if ctx.glucose < C.HYPO_L2 else "tăng đường huyết rất cao"
            # vẫn rà soát thuốc: sulfonylurea trên nền suy thận là nguyên nhân hạ đường huyết hay gặp
            return f"KHẨN CẤP ({kind})", ["glucose"] + (["medication"] if has_meds else [])
        steps = ["glucose", "nutrition"]
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
        findings = audit(ctx, results)

        seen, alerts = set(), []
        for r in results.values():
            for a in r.alerts:
                if a.message not in seen:
                    seen.add(a.message)
                    alerts.append(a)
        alerts.sort(key=lambda a: LEVEL_ORDER[a.level])
        sources = sorted({f"{e.source}: {e.title}" for r in results.values() for e in r.evidence})
        report = Report(plan, steps, ctx.classification, ctx.forecast, results, alerts,
                        [f"{f.rule}: {f.message}" for f in findings], None, None, sources, C.DISCLAIMER, readings)
        report.synthesis, report.synthesis_note = self._synthesize(report)
        return report

    def _synthesize(self, rep: Report) -> tuple[str | None, str | None]:
        if rep.plan.startswith("KHẨN"):
            return None, "Chế độ khẩn cấp: không dùng LLM, chỉ hiển thị hướng dẫn của các agent."
        if self.llm is None or not self.llm.enabled:
            return None, "Chưa cấu hình LLM (ANTHROPIC_API_KEY): hiển thị báo cáo của các agent."
        src = json.dumps({"phan_loai": rep.classification["class_name"], "nguy_co": rep.classification["risk_level"],
                          "canh_bao": [a.message for a in rep.alerts],
                          "agents": {k: {"tom_tat": r.summary, "viec_can_lam": r.actions} for k, r in rep.results.items()}},
                         ensure_ascii=False)
        text = self.llm.complete(SYNTH_SYSTEM, src)
        if not text:
            return None, f"LLM lỗi ({self.llm.last_error}): dùng báo cáo của các agent."
        ok, bad = is_grounded(text, src)
        if not ok:
            return None, f"Đã loại bản tổng hợp LLM vì chứa số liệu ngoài dữ liệu của agent ({', '.join(bad)})."
        return text, None
