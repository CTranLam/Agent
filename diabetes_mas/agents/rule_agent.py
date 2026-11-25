"""RuleAgent: QUYẾT ĐỊNH liều cuối bằng luật cố định, được giảm hoặc chặn đề xuất của RLAgent, không bao giờ tăng.

Luật R0-R6 và trần liều nằm ở policy/safety_rules.py. Liều bữa ăn do rule tính riêng (carb / ICR), RL không tham gia.
"""
from dataclasses import dataclass

from ..policy.safety_rules import apply_safety, cap_total_bolus, check_meal_bolus
from ..schemas import Alert, AnalysisContext


@dataclass
class DoseDecision:
    correction: float
    meal: float
    iob: float | None
    rules_fired: list[str]
    notes: list[str]
    blocked_reason: str | None
    alerts: list[Alert]


class RuleAgent:
    name, kind = "RuleAgent", "Rule"

    def decide(self, ctx: AnalysisContext, proposed: float) -> DoseDecision:
        p, g = ctx.profile, ctx.glucose
        dec = apply_safety(p, g, ctx.forecast_min, proposed, ctx.minutes_since_insulin, ctx.diabetes_type,
                           n_readings=ctx.n_readings, reading_age_min=ctx.reading_age_min,
                           last_insulin_units=ctx.last_insulin_units)
        # CHỈ lấy carb người dùng nhập; không suy ra từ thực đơn gợi ý
        meal, notes = check_meal_bolus(p, g, ctx.forecast_min, ctx.planned_carbs, reading_age_min=ctx.reading_age_min)
        correction, meal, cap_notes = cap_total_bolus(dec.dose, meal)
        return DoseDecision(correction, meal, dec.iob, dec.rules_fired, notes + cap_notes,
                            dec.rules_fired[0] if dec.blocked else None, dec.alerts)
