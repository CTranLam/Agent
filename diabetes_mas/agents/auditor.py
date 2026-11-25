"""Lưới an toàn độc lập: rà kết quả có cấu trúc của mọi agent. Vận hành bình thường phải không có phát hiện nào."""
from dataclasses import dataclass

from .. import config as C
from ..policy.exercise_gate import exercise_gate
from ..schemas import AgentResult, Alert, AnalysisContext

EXERCISE_WORDS = ("đi bộ", "vận động", "tập ", "đạp xe")


@dataclass
class Finding:
    rule: str
    message: str


def audit(ctx: AnalysisContext, results: dict[str, AgentResult]) -> list[Finding]:
    out: list[Finding] = []
    med = results.get("medication")

    def cancel(which: str, rule: str, why: str):
        med.details[which] = None if ctx.shared.get("review_only") else 0.0
        med.actions.append(f"Auditor: đã HUỶ liều {which} ({why}).")
        out.append(Finding(rule, f"huỷ liều {which}: {why}"))

    if med:
        d = med.details
        before = (d.get("correction"), d.get("meal"))
        correction, meal = d.get("correction") or 0, d.get("meal") or 0
        low_or_soon = ctx.glucose < C.HYPO_L1 or ctx.forecast_min < C.HYPO_L1
        stale = ctx.reading_age_min is not None and ctx.reading_age_min > C.READING_STALE_MIN
        if ctx.shared.get("review_only"):
            if correction > 0:
                cancel("correction", "A7", "chế độ khẩn cấp")
            if meal > 0:
                cancel("meal", "A7", "chế độ khẩn cấp")
        else:
            if correction > 0 and low_or_soon:
                cancel("correction", "A1", "đang/sắp hạ đường huyết")
            elif correction > 0 and (stale or ctx.n_readings < C.MIN_READINGS_FOR_DOSE):
                cancel("correction", "A3", "số đo cũ hoặc quá ít")
            elif correction > 0 and ctx.profile.on_insulin and ctx.minutes_since_insulin is None:
                cancel("correction", "A3", "chưa biết lần tiêm gần nhất")
            if meal > 0 and ctx.planned_carbs is None:
                cancel("meal", "A2", "người dùng không nhập lượng carb")
            elif meal > 0 and low_or_soon:
                cancel("meal", "A1", "đang/sắp hạ đường huyết")
            elif meal > 0 and stale:
                cancel("meal", "A3", "số đo cũ")
            correction, meal = d.get("correction") or 0, d.get("meal") or 0
            if correction > C.MAX_CORRECTION_DOSE:
                d["correction"] = correction = C.MAX_CORRECTION_DOSE
                out.append(Finding("A4", f"giảm liều bù về trần {C.MAX_CORRECTION_DOSE:g} đv"))
            if meal > C.MAX_MEAL_BOLUS:
                d["meal"] = meal = C.MAX_MEAL_BOLUS
                out.append(Finding("A4", f"giảm liều bữa ăn về trần {C.MAX_MEAL_BOLUS:g} đv"))
            if correction + meal > C.MAX_TOTAL_BOLUS:
                d["meal"] = meal = min(meal, C.MAX_TOTAL_BOLUS)
                d["correction"] = max(0.0, C.MAX_TOTAL_BOLUS - meal)
                out.append(Finding("A4", f"giảm tổng liều về trần {C.MAX_TOTAL_BOLUS:g} đv"))
            if not ctx.profile.on_insulin:
                if d.get("correction"):
                    cancel("correction", "A8", "bệnh nhân không dùng insulin")
                if d.get("meal"):
                    cancel("meal", "A8", "bệnh nhân không dùng insulin")
        if (d.get("correction"), d.get("meal")) != before:
            # văn bản phải khớp liều đã sửa
            med.actions[:] = [a for a in med.actions if not a.startswith(("Liều bù tham khảo", "Liều bữa ăn tham khảo"))]
            med.summary = f"Auditor đã điều chỉnh liều: bù {d.get('correction') or 0:g} đv, bữa ăn {d.get('meal') or 0:g} đv."

    gate = exercise_gate(ctx)
    if not gate.allowed:
        for key, r in results.items():
            if key == "exercise":
                continue
            kept = [a for a in r.actions if not any(w in a.lower() for w in EXERCISE_WORDS)]
            if len(kept) != len(r.actions):
                r.actions[:] = kept
                out.append(Finding("A5", f"gỡ lời khuyên vận động của {r.agent}: {gate.reason}"))

    if ctx.glucose < C.HYPO_L2 or ctx.glucose >= C.HYPER_CRITICAL:
        has_critical = any(a.level == "critical" for r in results.values() for a in r.alerts)
        if not has_critical and "glucose" in results:
            results["glucose"].alerts.insert(0, Alert("critical", f"Đường huyết ở mức nguy hiểm: {ctx.glucose:.0f} mg/dL. Liên hệ y tế ngay.", "Auditor"))
            out.append(Finding("A6", "bổ sung cảnh báo khẩn còn thiếu"))
    return out
