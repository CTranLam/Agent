"""Nguồn sự thật duy nhất về việc có được vận động hay không, để các agent không mâu thuẫn."""
from dataclasses import dataclass

from .. import config as C
from ..schemas import AnalysisContext


@dataclass
class ExerciseGate:
    allowed: bool
    intensity: str
    reason: str


def exercise_gate(ctx: AnalysisContext) -> ExerciseGate:
    p, g = ctx.profile, ctx.glucose
    if g < C.HYPO_L1 or ctx.forecast_min < C.HYPO_L1:
        return ExerciseGate(False, "không", "đang hoặc sắp hạ đường huyết")
    if g >= C.HYPER_SEVERE:
        return ExerciseGate(False, "không", f"đường huyết từ {C.HYPER_SEVERE} mg/dL: kiểm tra ceton và liên hệ nhóm điều trị trước")
    if g > C.HYPER_KETONE and (p.on_insulin or ctx.diabetes_type == "T1"):
        return ExerciseGate(False, "không", "đường huyết trên 250 mg/dL ở người dùng insulin/type 1: cần kiểm tra ceton trước")
    if g < 90:
        return ExerciseGate(True, "nhẹ", "đường huyết dưới 90 mg/dL: ăn nhẹ 15 g carbohydrate trước khi tập")
    if p.heart_failure or p.age >= 75:
        return ExerciseGate(True, "nhẹ", "suy tim hoặc từ 75 tuổi: chọn cường độ nhẹ")
    return ExerciseGate(True, "trung bình", "đường huyết trong khoảng an toàn để vận động (90-250 mg/dL)")
