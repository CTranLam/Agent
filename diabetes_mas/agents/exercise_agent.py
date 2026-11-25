from ..policy.exercise_gate import exercise_gate
from ..schemas import AgentResult, AnalysisContext
from .base import BaseAgent


class ExerciseAgent(BaseAgent):
    name, kind = "ExerciseAgent", "Rule + KB"

    def run(self, ctx: AnalysisContext) -> AgentResult:
        p, gate = ctx.profile, exercise_gate(ctx)
        details = {"allowed": gate.allowed}
        ev = self.rag.retrieve("an toàn đường huyết khi vận động", k=2, topic="exercise")
        if not gate.allowed:
            return AgentResult(self.name, f"Chưa nên vận động lúc này: {gate.reason}.", details,
                               [f"Hoãn tập cho đến khi đường huyết ổn định ({gate.reason})."], [], ev)
        actions = [gate.reason.capitalize() + "."]
        if gate.intensity == "trung bình":
            actions += ["Tối thiểu 150 phút/tuần cường độ trung bình (đi bộ nhanh, đạp xe), chia ≥3 ngày, không nghỉ quá 2 ngày liên tiếp.",
                        "Tập kháng lực 2-3 buổi/tuần, không liên tiếp."]
        else:
            actions.append("Bắt đầu nhẹ nhàng: đi bộ chậm 10-20 phút, tăng dần theo khả năng.")
        if p.on_insulin or ctx.diabetes_type == "T1":
            actions.append("Mang theo carbohydrate nhanh và đo đường huyết trước, sau khi tập.")
        if p.heart_disease or p.heart_failure:
            actions.append("Có bệnh tim mạch: nên được đánh giá trước khi tăng cường độ; dừng và gọi cấp cứu nếu đau ngực.")
        if p.pregnant:
            actions.append("Đang mang thai: hỏi bác sĩ sản khoa về mức tập phù hợp.")
        comp = ctx.classification.get("complications", {})
        if comp.get("retinopathy", 0) > 0.3 or comp.get("neuropathy", 0) > 0.3:
            actions.append("Nguy cơ biến chứng mắt/thần kinh cao hơn: tránh gắng sức mạnh và kiểm tra bàn chân sau khi tập.")
        return AgentResult(self.name, f"Có thể vận động cường độ {gate.intensity}.", details, actions, [], ev)
