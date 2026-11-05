from diabetes_mas.eval.report_metrics import build

RAG = {"n_questions": 1, "word_overlap": {"recall_at_3": .5, "mrr": .5}, "bm25": {"recall_at_3": 1, "mrr": 1}, "caveat": "c"}

def test_missing_artifacts_say_not_measured():
    t = build({}, RAG)
    assert t.count("Chưa đo") == 3 and "BM25" in t

def test_none_metric_renders():
    f = {k: {"mae": 1.0, "hypo_recall": None, "hypo_precision": None} for k in ("hold_last", "linear", "lstm", "conservative_min")}
    t = build({"forecast_metrics": {**f, "n_test_series": 1, "n_windows": 1, "n_hypo_windows": 0}}, RAG)
    assert "n/a" in t and "MÔ PHỎNG" in t
