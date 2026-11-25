import numpy as np

from .. import config as C
from ..policy.safety_rules import hyper_alerts
from ..schemas import AgentResult, Alert, AnalysisContext
from .base import BaseAgent


class GlucoseAgent(BaseAgent):
    name, kind = "GlucoseAgent", "Deep learning + Rule"

    def run(self, ctx: AnalysisContext) -> AgentResult:
        p, g, r = ctx.profile, ctx.glucose, np.asarray(ctx.readings, dtype=float)
        n = ctx.n_readings
        reliable = n >= C.MIN_READINGS_FOR_STATS  # với 1-2 số đo, "TIR 100%" là ảo tưởng
        details = {
            "current": round(g, 1),
            "cv_pct": round(float(r.std() / r.mean() * 100), 1) if reliable else None,
            "tir_pct": round(float(((r >= 70) & (r <= 180)).mean() * 100), 1) if reliable else None,
            "tbr_pct": round(float((r < 70).mean() * 100), 1) if reliable else None,
            "tar_pct": round(float((r > 180).mean() * 100), 1) if reliable else None,
        }
        alerts: list[Alert] = []
        actions: list[str] = []

        if not reliable:
            alerts.append(Alert("info", f"Chất lượng dữ liệu: chỉ có {n} số đo (<{C.MIN_READINGS_FOR_STATS}, tức <1 giờ). "
                                        "Xu hướng và dự báo kém tin cậy - không thể loại trừ hạ đường huyết sắp tới; "
                                        "hãy đo lại nếu có triệu chứng."))
        if ctx.reading_age_min is not None and ctx.reading_age_min > C.READING_STALE_MIN:
            alerts.append(Alert("warning", f"Số đo đã cũ {int(ctx.reading_age_min)} phút: hãy đo lại."))
        if n >= 2 and np.abs(np.diff(r)).max() > 30:
            alerts.append(Alert("warning", "Đường huyết thay đổi hơn 30 mg/dL trong 5 phút: nghi lỗi cảm biến, hãy đo lại bằng đầu ngón tay."))

        if g < C.HYPO_L1:
            level = "critical" if g < C.HYPO_L2 else "warning"
            alerts.append(Alert(level, f"Hạ đường huyết {'mức 2' if g < C.HYPO_L2 else 'mức 1'}: {g:.0f} mg/dL."))
            foods = [f"{f['name']} ({f['portion']})" for f in self.kb.foods if f["category"] == "fast_carb"]
            actions.append(f"Dùng ngay 15 g carbohydrate nhanh: {', '.join(foods)}; đo lại sau 15 phút, lặp lại nếu vẫn dưới 70.")
            if g < C.HYPO_L2:
                actions.append("Nếu lơ mơ hoặc không nuốt được: không cho ăn uống, gọi cấp cứu, dùng glucagon nếu có.")
            state, query, topic = "dưới mục tiêu", "xử trí hạ đường huyết", "hypoglycemia"
        elif g > 180:
            alerts += [Alert(a.level, a.message, self.name) for a in hyper_alerts(g, p.on_insulin, ctx.diabetes_type)]
            actions.append("Uống đủ nước, kiểm tra đã dùng thuốc đúng chưa và đo lại sau 1-2 giờ.")
            state, query, topic = "trên mục tiêu", "xử trí tăng đường huyết ceton", "hyperglycemia"
            if ctx.forecast_min < C.HYPO_L1:
                alerts.append(Alert("warning", f"Dự báo đường huyết có thể xuống {ctx.forecast_min:.0f} mg/dL trong 30 phút tới."))
        elif ctx.forecast_min < C.HYPO_L1:
            alerts.append(Alert("warning", f"Dự báo đường huyết có thể xuống {ctx.forecast_min:.0f} mg/dL trong 30 phút tới."))
            actions.append("Chuẩn bị carbohydrate nhanh và đo lại sau 15 phút.")
            state, query, topic = "trong khoảng mục tiêu nhưng đang giảm", "xử trí hạ đường huyết", "hypoglycemia"
        else:
            actions.append("Đường huyết đang trong khoảng mục tiêu - tiếp tục duy trì thói quen hiện tại.")
            state, query, topic = "trong khoảng mục tiêu", "mục tiêu thời gian trong khoảng", "hyperglycemia"

        f = ctx.forecast
        summary = (f"Đường huyết hiện tại {g:.0f} mg/dL ({state}), xu hướng {f['trend_per_5min']:+.1f} mg/dL/5 phút, "
                   f"dự báo sau 30 phút: {f['values'][-1]:.0f} mg/dL (ước lượng bảo thủ thấp nhất {ctx.forecast_min:.0f}).")
        if not reliable:
            summary += f" Chưa tính TIR/TBR/CV vì chỉ có {n} số đo (cần ≥{C.MIN_READINGS_FOR_STATS})."
        for a in alerts:
            a.agent = self.name
        return AgentResult(self.name, summary, details, actions, alerts, self.rag.retrieve(query, k=2, topic=topic))
