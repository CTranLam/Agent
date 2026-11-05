import datetime as dt
import random
import zlib

from .. import config as C
from ..kb.store import fold
from ..policy.exercise_gate import exercise_gate
from ..schemas import AgentResult, Alert, AnalysisContext, PatientProfile
from .base import BaseAgent

MEAL_LABEL = {"breakfast": "sáng", "lunch": "trưa", "dinner": "tối"}
MEAL_ALIAS = {"sang": "breakfast", "trua": "lunch", "toi": "dinner"}
PORTIONS = (0.5, 1.0, 1.5, 2.0)
ALLERGY_GROUPS = {"hai san": {"tom", "ca"}, "seafood": {"tom", "ca"}, "lua mi": {"gluten"}, "bot mi": {"gluten"}}


def energy_kcal(p: PatientProfile, dtype: str) -> tuple[float, str]:
    """Mifflin-St Jeor x hệ số hoạt động nhẹ 1,3."""
    male = str(p.sex).upper().startswith("M")
    bmr = 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age + (5 if male else -161)
    tdee, note = bmr * 1.3, "duy trì cân nặng"
    if p.bmi >= 23 and dtype != "T1" and not (p.age >= 75 or p.bmi < 18.5) and not p.pregnant:
        tdee, note = max(1500 if male else 1200, tdee - 500), "giảm ~500 kcal để giảm cân (BMI ≥23)"
    return round(tdee, -1), note


def carb_per_meal(p: PatientProfile, dtype: str) -> tuple[float, dict]:
    kcal, note = energy_kcal(p, dtype)
    daily = kcal * C.CARB_SHARE / 4
    lo = 45.0 if (p.age >= 75 or p.bmi < 18.5 or p.pregnant) else 30.0
    per = max(lo, min(60.0, round((daily - C.SNACK_CARB_G) / 3 / 5) * 5))
    return per, {"energy_kcal": kcal, "energy_note": note, "daily_carb_g": round(daily)}


class NutritionAgent(BaseAgent):
    name = "Dinh dưỡng"

    def _banned(self, allergy: str) -> set[str]:
        """Dị ứng người dùng gõ -> các chất gây dị ứng trong KB ("trứng gà" -> trứng, "hải sản" -> tôm, cá)."""
        a = fold(allergy).strip()
        words = set(a.split())
        known = {fold(x) for f in self.kb.foods for x in f["allergens"]}
        return ALLERGY_GROUPS.get(a, set()) | {k for k in known if words and (set(k.split()) <= words or words <= set(k.split()))}

    def _allowed(self, p: PatientProfile) -> list[dict]:
        banned = set().union(*[self._banned(a) for a in p.allergies]) if p.allergies else set()
        low_salt = p.hypertension or p.heart_failure or p.egfr < 60
        return [f for f in self.kb.foods
                if not ({fold(a) for a in f["allergens"]} & banned)
                and (f["vegetarian"] or not p.vegetarian)
                and not (low_salt and f["high_sodium"])]

    def run(self, ctx: AnalysisContext) -> AgentResult:
        p, dtype = ctx.profile, ctx.diabetes_type
        budget, info = carb_per_meal(p, dtype)
        meal = MEAL_ALIAS.get(ctx.meal_type, ctx.meal_type)
        if meal not in MEAL_LABEL:
            meal = "lunch"
        # hạt giống từ hồ sơ + ngày: cùng người cùng ngày cùng thực đơn, ngày khác đổi món
        seed = zlib.crc32(f"{p.name}|{p.age}|{p.weight_kg}|{meal}|{dt.date.today()}".encode())
        rng = random.Random(seed)
        foods = self._allowed(p)
        pick = lambda cat: [f for f in foods if f["category"] == cat and meal in f["meals"]]

        veg = rng.sample(pick("vegetable"), min(2, len(pick("vegetable"))))
        protein = rng.sample(pick("protein"), min(1, len(pick("protein"))))
        items = [{"name": f["name"], "portion": f"1 × {f['portion']}", "carb_g": f["carb_g"]} for f in protein + veg]
        used = sum(i["carb_g"] for i in items)
        starches = [f for f in pick("starch") if f["gi"] != "cao"] or pick("starch")
        if starches:
            s = rng.choice(starches)
            fits = [m for m in PORTIONS if s["carb_g"] * m + used <= budget]
            if fits:
                m = fits[-1]
                items.insert(0, {"name": s["name"], "portion": f"{m:g} × {s['portion']}", "carb_g": round(s["carb_g"] * m)})
        total = sum(i["carb_g"] for i in items)
        snack_foods = [f for f in foods if "snack" in f["meals"]]
        snacks = [{"name": f["name"], "portion": f["portion"], "carb_g": f["carb_g"]}
                  for f in rng.sample(snack_foods, min(2, len(snack_foods)))]

        actions = [f"Bữa {MEAL_LABEL[meal]}: khoảng {total} g carb (ngân sách tham khảo {budget:.0f} g; ~{info['energy_kcal']:.0f} kcal/ngày - "
                   f"{info['energy_note']}), ưu tiên GI thấp-trung bình."]
        if ctx.glucose < C.HYPO_L1 or ctx.forecast_min < C.HYPO_L1:
            actions.insert(0, "Đang hoặc sắp hạ đường huyết: xử trí bằng carbohydrate nhanh trước, chưa áp dụng thực đơn này.")
        if exercise_gate(ctx).allowed:
            actions.append("Đi bộ nhẹ 10-15 phút sau ăn để giảm đường huyết sau ăn.")
        actions.append("Tránh nước ngọt có đường, trà sữa, nước ép đóng chai (trừ khi xử trí hạ đường huyết).")
        if p.hypertension or p.heart_failure or p.egfr < 60:
            actions.append("Hạn chế muối (<5 g/ngày): giảm nước chấm, đồ muối, đồ chế biến sẵn.")
        if p.on_insulin and ctx.planned_carbs is None:
            actions.append("Đang dùng insulin: hãy nhập lượng carb thực tế định ăn để có liều bữa ăn tham khảo.")

        details = {"budget_g": budget, **info, "meal": meal,
                   "menu": {"items": items, "total_carb": total, "snacks": snacks}}
        summary = f"Gợi ý bữa {MEAL_LABEL[meal]} ~{total} g carbohydrate (ngân sách {budget:.0f} g); thực đơn lấy từ cơ sở tri thức."
        alerts = [Alert("warning", f"Chưa đối chiếu được dị ứng '{a}' với thực đơn: hãy tự kiểm tra từng món.", self.name)
                  for a in p.allergies if a.strip() and not self._banned(a)]
        return AgentResult(self.name, summary, details, actions, alerts,
                           self.rag.retrieve("nguyên tắc dinh dưỡng tinh bột", k=2, topic="nutrition"))
