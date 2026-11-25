"""Sinh docs/metrics.md từ kết quả đo thật: python -m diabetes_mas.eval.report_metrics"""
import json
from pathlib import Path

from .rag_eval import evaluate as rag_evaluate


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def build(meta: dict, rag: dict) -> str:
    parts = ["# Kết quả đo\n\nSinh tự động bởi `python -m diabetes_mas.eval.report_metrics`.\n"]

    parts.append("## Phân loại (DiabetesNet)\n")
    if "metrics" in meta:
        m, b = meta["metrics"], meta["baselines"]
        parts.append(f"{meta['n_rows_raw']:,} dòng, còn {meta['n_rows_dedup']:,} sau khi loại trùng; ngưỡng {meta['threshold']:.3f} chọn trên validation cho recall ≥ 90%. Số đo trên tập test.\n")
        parts.append(_table(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
                            [[_f(m[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")]]))
        parts.append("\n" + _table(["Mô hình", "ROC-AUC"], [["Chỉ HbA1c", _f(b["hba1c_only_roc_auc"])], ["Chỉ đường huyết", _f(b["glucose_only_roc_auc"])],
                                                          ["Hồi quy logistic 13 đặc trưng", _f(b["logistic_roc_auc"])], ["DiabetesNet", _f(m["roc_auc"])]]))
        parts.append("\n**Giới hạn:** nhãn dataset gần như do hai ngưỡng HbA1c và đường huyết quyết định, nên AUC cao không phải khả năng dự báo trên người thật. Mạng sâu chỉ hơn logistic một chút.\n")
    else:
        parts.append("Chưa đo: chạy `python -m diabetes_mas.dl.train`.\n")

    parts.append("## Dự báo đường huyết 30 phút\n")
    if "forecast_metrics" in meta:
        f = meta["forecast_metrics"]
        names = [("hold_last", "Giữ nguyên giá trị cuối"), ("linear", "Ngoại suy tuyến tính"), ("lstm", "GlucoseLSTM"), ("conservative_min", "min(LSTM, tuyến tính), hệ thống dùng")]
        parts.append(f"{f['n_test_series']} chuỗi test, {f['n_windows']:,} cửa sổ, {f['n_hypo_windows']} cửa sổ có hạ đường huyết sắp xảy ra.\n")
        parts.append(_table(["Phương pháp", "MAE (mg/dL)", "Recall hạ đường huyết", "Precision"],
                            [[n, _f(f[k]["mae"], 2), _f(f[k]["hypo_recall"]), _f(f[k]["hypo_precision"])] for k, n in names]))
        parts.append("\n**Giới hạn:** CGM MÔ PHỎNG do chính dự án tạo, các cửa sổ cùng chuỗi còn tương quan; chưa kiểm chứng trên người thật.\n")
    else:
        parts.append("Chưa đo: chạy `python -m diabetes_mas.dl.train_lstm`.\n")

    parts.append("## Học tăng cường đề xuất liều\n")
    if "rl_metrics" in meta:
        r = meta["rl_metrics"]
        names = [("no_insulin", "Không tiêm"), ("formula", "Công thức (G-110)/ISF"), ("formula_trend", "Công thức có xét xu hướng"), ("rl", "RL (bảng Q)")]
        parts.append(_table(["Chính sách", "TIR", "Hạ đường huyết do liều", "Trên 180"],
                            [[n, f"{r[k]['tir']:.1%}", f"{r[k]['dose_hypo']:.2%}", f"{r[k]['above_180']:.1%}"] for k, n in names]))
        parts.append(f"\n**Giới hạn:** {r['data']}. RL không vượt công thức có xét xu hướng; giá trị của lớp này là kiến trúc RL đề xuất rồi rule phủ quyết.\n")
    else:
        parts.append("Chưa đo: chạy `python -m diabetes_mas.policy.rl_insulin --eval`.\n")

    parts.append("## RAG\n")
    rows = [["Đếm từ trùng", rag["word_overlap"]["recall_at_3"], rag["word_overlap"]["mrr"]],
            ["BM25 + bigram + đồng nghĩa", rag["bm25"]["recall_at_3"], rag["bm25"]["mrr"]]]
    if "vector" in rag:
        rows.append([f"Vector: {rag['vector']['embedder']}, cosine", rag["vector"]["recall_at_3"], rag["vector"]["mrr"]])
    if "hybrid" in rag:
        rows.append(["Lai BM25 + vector (RRF)", rag["hybrid"]["recall_at_3"], rag["hybrid"]["mrr"]])
    parts.append(_table(["Bộ truy xuất", "Recall@3", "MRR"], rows))
    if "vector" in rag:
        parts.append("\nHệ thống mặc định dùng BM25 (`RAG_RETRIEVER=bm25`); đổi sang `vector` hoặc `hybrid` qua biến môi trường. "
                     "Vector LSA học từ chính vài chục đoạn trong KB nên chưa hiểu đồng nghĩa thật; embedding từ mô hình ngôn ngữ "
                     "(`EMBED_BASE_URL`, `EMBED_MODEL`) sẽ cho số khác, cần đo lại bằng lệnh trên.\n")
    parts.append(f"\n{rag['n_questions']} câu hỏi. **Giới hạn:** {rag['caveat']}\n")
    return "\n".join(parts)


if __name__ == "__main__":
    p = Path("artifacts/meta.json")
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    Path("docs").mkdir(exist_ok=True)
    Path("docs/metrics.md").write_text(build(meta, rag_evaluate()), encoding="utf-8")
    print("Đã ghi docs/metrics.md")
