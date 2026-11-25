"""Rà soát thuốc (rule + KB) và điều phối liều insulin: RLAgent đề xuất -> RuleAgent quyết định."""
import re

from ..schemas import AgentResult, Alert, AnalysisContext
from .base import BaseAgent
from .conditions import evaluate
from .rl_agent import RLAgent
from .rule_agent import RuleAgent


class MedicationAgent(BaseAgent):
    name, kind = "MedicationAgent", "RL + Rule + KB"

    def __init__(self, kb, rag, policy):
        super().__init__(kb, rag)
        self.rl, self.rule = RLAgent(policy), RuleAgent()

    def _review(self, ctx: AnalysisContext):
        p, dtype = ctx.profile, ctx.diabetes_type
        alerts, actions, known, unknown = [], [], [], []
        entries = [x.strip() for raw in p.medications for x in re.split(r"[;,+/]|\s+và\s+", raw, flags=re.I)]
        for raw in filter(None, entries):
            d = self.kb.match_drug(raw)
            if d is None:
                unknown.append(raw)
                alerts.append(Alert("warning", f"Chưa nhận ra thuốc '{raw}' trong cơ sở tri thức: cần dược sĩ kiểm tra.", self.name))
            elif d not in known:
                known.append(d)
        if known and dtype in ("NONE", "PRE"):
            alerts.append(Alert("warning", "Đang dùng thuốc tiểu đường nhưng chỉ số chưa đạt tiêu chuẩn chẩn đoán: hãy chọn loại tiểu đường trong hồ sơ.", self.name))
        names = [d["name"] for d in known]

        for d in known:
            name, title = d["name"], d["name"].capitalize()
            for code, label, level in [(c, "chống chỉ định", "critical") for c in d["contraindications"]] \
                    + [(c, "thận trọng", "warning") for c in d["cautions"]]:
                hit = evaluate(code, p, dtype)
                if hit is None:
                    alerts.append(Alert("warning", f"{title}: điều kiện '{code}' chưa đánh giá tự động được, cần dược sĩ kiểm tra.", self.name))
                elif hit:
                    alerts.append(Alert(level, f"{title}: {label} - {d['notes'].get(code, code)}.", self.name))
                    for alt in self.kb.alternatives(name, code):
                        cand = next((x for x in self.kb.drugs if x["name"] == alt["alternative"]), None)
                        blocked = cand is not None and (dtype in cand["not_for_types"] or any(
                            evaluate(c, p, dtype) for c in cand["contraindications"] + cand["cautions"]))
                        line = f"Trao đổi với bác sĩ về việc thay {name} bằng {alt['alternative']} ({alt['why']})."
                        if not blocked and alt["alternative"] not in names and line not in actions:
                            actions.append(line)
            if dtype in d["not_for_types"]:
                alerts.append(Alert("warning", f"{title}: không phù hợp với chẩn đoán hiện tại ({dtype}), cần bác sĩ xem lại.", self.name))
        for i in self.kb.interactions(names):
            alerts.append(Alert(i["severity"], i["message"], self.name))
        return alerts, actions, names, unknown

    def run(self, ctx: AnalysisContext) -> AgentResult:
        p, dtype = ctx.profile, ctx.diabetes_type
        alerts, actions, names, unknown = self._review(ctx)
        correction, meal = 0.0, 0.0
        proposed, rl_state, iob, rules, notes, blocked_reason = None, None, None, [], [], None

        if ctx.shared.get("review_only"):
            correction, meal = None, None
            actions.append("Chế độ khẩn cấp: hệ thống KHÔNG đề xuất bất kỳ liều insulin nào - xử trí theo hướng dẫn khẩn cấp và liên hệ y tế.")
        elif p.on_insulin and dtype in ("T1", "T2"):
            proposed, rl_state = self.rl.propose(ctx), self.rl.state(ctx)
            dec = self.rule.decide(ctx, proposed)
            correction, meal, iob, rules, notes = dec.correction, dec.meal, dec.iob, dec.rules_fired, dec.notes
            blocked_reason = dec.blocked_reason
            if blocked_reason:  # không hiện số RL đề xuất cho ca đã bị chặn
                proposed = rl_state = None
            alerts += [Alert(a.level, a.message, self.name) for a in dec.alerts]
            actions += notes
            if correction > 0:
                actions.append(f"Liều bù tham khảo: {correction:g} đv (RL đề xuất {proposed:g} đv, sau rule an toàn).")
            if meal > 0:
                actions.append(f"Liều bữa ăn tham khảo: {meal:g} đv ({ctx.planned_carbs:g} g carb / ICR {p.icr:g}).")
        actions.append("Không tự ý đổi hoặc ngừng thuốc; mọi gợi ý chỉ để trao đổi với bác sĩ, dược sĩ.")

        details = {"drugs": names, "unknown_drugs": unknown, "proposed": proposed, "rl": rl_state, "blocked_reason": blocked_reason, "correction": correction,
                   "meal": meal, "iob": iob, "rules_fired": rules, "notes": notes}
        summary = f"Rà soát {len(names)} thuốc" + (f", {len(unknown)} thuốc chưa nhận ra" if unknown else "") + "."
        if correction is None:
            summary += " Không đề xuất liều trong chế độ khẩn cấp."
        elif blocked_reason:
            summary += f" Không đề xuất liều bù ({blocked_reason})."
            if meal > 0:
                summary += f" Liều bữa ăn tham khảo: {meal:g} đv."
        elif p.on_insulin:
            summary += f" Liều tham khảo: bù {correction:g} đv, bữa ăn {meal:g} đv."
        query = " ".join(names) if names else "an toàn khi dùng insulin"
        return AgentResult(self.name, summary, details, actions, alerts,
                           self.rag.retrieve(query + " thuốc", k=2, topic="medication"))
