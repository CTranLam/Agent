"""Rule an toàn có quyền phủ quyết. Nguyên tắc: không biết thì không tiêm."""
import math
from dataclasses import dataclass, field

from .. import config as C
from ..schemas import Alert, PatientProfile


@dataclass
class SafetyDecision:
    dose: float
    proposed: float
    iob: float | None = None
    blocked: bool = False
    rules_fired: list[str] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)


def floor_half(x: float) -> float:
    """Làm tròn XUỐNG bội 0,5 để không bao giờ vượt trần."""
    return math.floor(x * 2 + 1e-9) / 2


def iob_fraction(minutes: float, dia: float = C.INSULIN_ACTION_MIN) -> float:
    """Tỉ lệ liều còn hoạt động: đường cong trơn 1 -> 0 (xấp xỉ, không phải mô hình dược động học)."""
    if minutes <= 0:
        return 1.0
    if minutes >= dia:
        return 0.0
    x = minutes / dia
    return 1.0 - (3 * x * x - 2 * x ** 3)


def apply_safety(profile: PatientProfile, glucose: float, forecast_min: float, proposed: float,
                 minutes_since_insulin: float | None, dtype: str, *, n_readings: int | None = None,
                 reading_age_min: float | None = None, last_insulin_units: float | None = None) -> SafetyDecision:
    d = SafetyDecision(dose=proposed, proposed=proposed)

    def block(rule: str, level: str | None = None, msg: str | None = None):
        d.blocked, d.dose = True, 0.0
        d.rules_fired.append(rule)
        if msg:
            d.alerts.append(Alert(level, msg))

    if glucose < C.HYPO_L2:
        block("R1: glucose < 54 -> cấm insulin", "critical",
              "Hạ đường huyết mức 2: KHÔNG dùng insulin; xử trí ngay bằng carbohydrate nhanh.")
    elif glucose < C.HYPO_L1:
        block("R2: glucose < 70 -> cấm insulin", "warning", "Hạ đường huyết: không dùng insulin bù.")
    elif forecast_min < C.HYPO_L1:
        block("R3: dự báo < 70 trong 30 phút -> cấm insulin", "warning",
              "Dự báo đường huyết sẽ dưới 70 mg/dL trong 30 phút tới: hoãn mọi liều insulin bù.")
    if not profile.on_insulin and not d.blocked:
        block("R4: bệnh nhân không dùng insulin -> không đề xuất liều")
    if not d.blocked and reading_age_min is not None and reading_age_min > C.READING_STALE_MIN:
        block("R0a: số đo cũ -> không đề xuất liều", "warning",
              f"Số đo đã cũ {int(reading_age_min)} phút: hãy đo lại trước khi cân nhắc liều.")
    if not d.blocked and (n_readings is None or n_readings < C.MIN_READINGS_FOR_DOSE):
        block("R0b: quá ít số đo -> chưa biết xu hướng", "info",
              f"Cần ít nhất {C.MIN_READINGS_FOR_DOSE} số đo liên tiếp để cân nhắc liều bù.")

    if not d.blocked and proposed > 0 and minutes_since_insulin is not None \
            and minutes_since_insulin < C.INSULIN_ACTION_MIN:
        if last_insulin_units is None:
            block("R5: insulin trước còn tác dụng, không rõ liều -> chặn", "warning",
                  f"Mới tiêm insulin cách đây {int(minutes_since_insulin)} phút (chưa rõ liều): "
                  "không bù thêm để tránh hạ đường huyết do cộng dồn.")
        else:
            d.iob = round(last_insulin_units * iob_fraction(minutes_since_insulin), 2)
            d.dose = max(0.0, d.dose - d.iob)
            d.rules_fired.append(f"R5: trừ insulin còn hoạt động {d.iob} đv")

    if not d.blocked and d.dose > 0:
        cap = max(0.0, (glucose - C.TARGET_GLUCOSE) / max(profile.isf, 1.0) - (d.iob or 0.0))
        capped = floor_half(min(d.dose, C.MAX_CORRECTION_DOSE, cap))
        if capped < d.dose:
            d.rules_fired.append(f"R6: giới hạn liều còn {capped} đv")
        d.dose = capped
    if d.dose > 0:
        d.alerts.append(Alert("info", "Liều chỉ mang tính THAM KHẢO: xác nhận với bác sĩ, không tự chỉnh liều."))
    return d


def check_meal_bolus(profile: PatientProfile, glucose: float, forecast_min: float, carbs: float | None, *,
                     reading_age_min: float | None = None) -> tuple[float, list[str]]:
    """Liều bữa ăn = carb / ICR. CHỈ tính khi người dùng tự nhập carb."""
    notes: list[str] = []
    if not profile.on_insulin:
        return 0.0, notes
    if carbs is None:
        notes.append("Chưa tính liều bữa ăn: hãy nhập lượng carb dự định ăn (g) để có liều tham khảo.")
        return 0.0, notes
    if glucose < C.HYPO_L1 or forecast_min < C.HYPO_L1:
        notes.append("Chưa tính liều bữa ăn: xử trí/kiểm tra lại đường huyết trước (đang thấp hoặc sắp thấp).")
        return 0.0, notes
    if reading_age_min is not None and reading_age_min > C.READING_STALE_MIN:
        notes.append("Chưa tính liều bữa ăn: số đo đã cũ, hãy đo lại.")
        return 0.0, notes
    dose = floor_half(max(carbs, 0.0) / max(profile.icr, 1.0))
    if dose > C.MAX_MEAL_BOLUS:
        notes.append(f"Liều bữa ăn bị giới hạn tối đa {C.MAX_MEAL_BOLUS:g} đv - hãy xác nhận với bác sĩ.")
        dose = C.MAX_MEAL_BOLUS
    return dose, notes


def cap_total_bolus(correction: float, meal: float) -> tuple[float, float, list[str]]:
    """Tổng không quá trần; giảm liều bù trước vì bữa ăn còn cần insulin thật."""
    if correction + meal <= C.MAX_TOTAL_BOLUS:
        return correction, meal, []
    meal2 = min(meal, C.MAX_TOTAL_BOLUS)
    corr2 = max(0.0, C.MAX_TOTAL_BOLUS - meal2)
    return corr2, meal2, [f"Tổng liều bị giới hạn tối đa {C.MAX_TOTAL_BOLUS:g} đv."]


def hyper_alerts(glucose: float, on_insulin: bool, dtype: str) -> list[Alert]:
    """Độc lập với insulin: người chỉ dùng thuốc uống vẫn cần cảnh báo."""
    out = []
    if glucose > C.HYPER_KETONE and (on_insulin or dtype == "T1"):
        out.append(Alert("warning", "Đường huyết trên 250 mg/dL: hãy kiểm tra ceton."))
    if glucose >= C.HYPER_CRITICAL:
        out.append(Alert("critical", "Đường huyết rất cao (từ 400 mg/dL): liên hệ cơ sở y tế ngay."))
    elif glucose >= C.HYPER_SEVERE:
        out.append(Alert("warning", "Đường huyết từ 300 mg/dL: liên hệ nhóm điều trị trong ngày."))
    return out
