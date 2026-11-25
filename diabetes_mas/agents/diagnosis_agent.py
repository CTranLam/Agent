"""Nhận kết quả DiabetesNet và tiêu chuẩn ADA, giải thích kết luận và nêu việc cần làm để xác nhận."""
from ..dl.decision import meets_ada_diabetes
from ..schemas import AgentResult, Alert, AnalysisContext, PatientProfile
from .base import BaseAgent


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def _num(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _criteria(p: PatientProfile) -> list[str]:
    """Chỉ tính số người dùng nhập; đường huyết đói ước lượng không được dùng để chẩn đoán."""
    out = []
    if p.hba1c >= 6.5:
        out.append(f"HbA1c {_num(p.hba1c)}% (từ 6,5%)")
    if p.fasting_glucose is not None and p.fasting_glucose >= 126:
        out.append(f"Đường huyết đói {_num(p.fasting_glucose)} mg/dL (từ 126 mg/dL)")
    return out


class DiagnosisAgent(BaseAgent):
    name, kind = "DiagnosisAgent", "Deep learning + Rule"

    def run(self, ctx: AnalysisContext) -> AgentResult:
        p, c = ctx.profile, ctx.classification
        code, prob, thr = c["code"], c["prob"], c["threshold"]
        known = bool(p.diabetes_type_known)
        screened, ada = prob >= thr, meets_ada_diabetes(p)
        inferred = code in ("T1", "T2") and not known

        missing = []
        if not known:
            if p.fasting_glucose is None:
                missing.append("đường huyết đói")
            if inferred:
                if p.c_peptide is None:
                    missing.append("C-peptide")
                if p.autoantibody is None:
                    missing.append("kháng thể tự miễn")

        actions, alerts = [], []
        if known:
            actions.append(f"Đã có chẩn đoán {c['class_name']}: giữ nguyên, mô hình không thay đổi chẩn đoán của bác sĩ.")
        else:
            if ada:
                actions.append("Chỉ số đạt tiêu chuẩn chẩn đoán: cần xét nghiệm nhắc lại để bác sĩ xác nhận.")
                if not screened:
                    alerts.append(Alert("warning", "Mô hình dưới ngưỡng nhưng chỉ số đạt tiêu chuẩn ADA: ưu tiên tiêu chuẩn và đi xét nghiệm xác nhận.", self.name))
            elif p.on_insulin:
                actions.append("Đang dùng insulin nhưng chưa nhập loại tiểu đường: hãy chọn loại trong hồ sơ.")
            elif code == "PRE":
                actions.append("Chỉ số ở mức tiền tiểu đường: nên xét nghiệm nhắc lại để xác nhận.")
            elif screened:
                actions.append("Mô hình báo nguy cơ cao dù chỉ số chưa đạt tiêu chuẩn: nên đi xét nghiệm HbA1c và đường huyết đói.")
            else:
                actions.append("Chưa có dấu hiệu tiểu đường theo dữ liệu đã nhập: tiếp tục kiểm tra sức khoẻ định kỳ.")
            if screened and not ada and not p.on_insulin:
                alerts.append(Alert("info", "Mô hình vượt ngưỡng sàng lọc nhưng chỉ số chưa đạt tiêu chuẩn: đây là nguy cơ, không phải chẩn đoán.", self.name))
            if missing:
                actions.append("Xét nghiệm còn thiếu: " + ", ".join(missing) + ".")

        state = "vượt" if screened else "chưa vượt"
        summary = (f"Mạng DiabetesNet cho xác suất {_pct(prob)}, {state} ngưỡng sàng lọc {_pct(thr)}. "
                   f"Kết luận: {c['class_name']} (căn cứ: {c['source']}). Kết quả này cần bác sĩ xác nhận.")
        details = {"code": code, "class_name": c["class_name"], "prob": prob, "threshold": thr, "screened": screened,
                   "meets_ada": ada, "criteria": _criteria(p), "source": c["source"], "inferred_type": inferred,
                   "missing_tests": missing}
        return AgentResult(self.name, summary, details, actions, alerts,
                           self.rag.retrieve("tiêu chuẩn chẩn đoán tiền đái tháo đường HbA1c", k=2, topic="hyperglycemia"))
