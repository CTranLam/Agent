"""RLAgent: chỉ ĐỀ XUẤT liều insulin bù, không có quyền quyết định (xem rule_agent.py).

Trạng thái: đường huyết, xu hướng mỗi 5 phút, ISF. Hành động: 0-4 đv. Chính sách: bảng Q học trước
trên simulator (policy/rl_insulin.py), không học thêm khi chạy.
"""
from ..schemas import AnalysisContext


class RLAgent:
    name, kind = "RLAgent", "RL"

    def __init__(self, policy):
        self.policy = policy

    def state(self, ctx: AnalysisContext) -> dict:
        return {"glucose": ctx.glucose, "trend_per_5min": ctx.forecast["trend_per_5min"], "isf": ctx.profile.isf}

    def propose(self, ctx: AnalysisContext) -> float:
        s = self.state(ctx)
        return self.policy.suggest(s["glucose"], s["trend_per_5min"], s["isf"]) if s["isf"] else 0.0
