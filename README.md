# Hệ đa agent hỗ trợ quản lý tiểu đường

Demo môn học Phát triển các hệ thống thông minh. Là công cụ hỗ trợ học tập, **không phải thiết bị y tế, không thay thế bác sĩ**; cơ sở tri thức là bản tóm tắt giáo dục chưa được thẩm định.

## Sơ đồ kiến trúc và mã nguồn

| Khối trên sơ đồ | Hiện thực |
|---|---|
| DB có sẵn | `data/diabetes.csv`, CGM mô phỏng `dl/synthetic.py` |
| Deep learning → Phân loại | `DiabetesNet` (`dl/models.py`, `dl/train.py`) + tiêu chuẩn ADA (`dl/decision.py`) |
| Deep learning → Dự đoán | `GlucoseLSTM` (`dl/train_lstm.py`), dự báo bảo thủ |
| Web, sách → KB | `knowledge_docs/*.md`, `kb/foods.json`, `kb/drugs.json`, `kb/store.py` |
| RAG | `kb/rag.py` (BM25 + bigram + đồng nghĩa), trả về đoạn trích kèm nguồn |
| Reinforcement learning | `policy/rl_insulin.py` |
| Rule | `policy/safety_rules.py`, `policy/exercise_gate.py`, `agents/auditor.py` |
| Agent | `agents/orchestrator.py` + 4 agent chuyên trách |
| Tư vấn, trợ giúp | `api.py`, `web/index.html` |

## Cài và chạy

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn diabetes_mas.api:app --host 127.0.0.1 --port 8000   # mở http://127.0.0.1:8000
```

Model đã huấn luyện có sẵn trong `artifacts/`. Hệ thống không gọi API ngoài và không dùng LLM: mô hình học sâu và RL do dự án tự huấn luyện, phần hỏi đáp trả đoạn trích từ cơ sở tri thức.

## Huấn luyện lại và đo

Tải "Diabetes prediction dataset" (Kaggle, hoặc bản sao trên Hugging Face `GB2024/diabetes`) về `data/diabetes.csv`, rồi:

```bash
.venv/bin/python -m diabetes_mas.dl.train
.venv/bin/python -m diabetes_mas.dl.train_lstm
.venv/bin/python -m diabetes_mas.policy.rl_insulin
.venv/bin/python -m diabetes_mas.policy.rl_insulin --eval
.venv/bin/python -m diabetes_mas.eval.report_metrics   # ghi docs/metrics.md
.venv/bin/pytest -q
```

## Giới hạn

- Nhãn dataset gần như do hai ngưỡng quyết định; không có nhãn type 1/2 nên type do luật suy ra.
- CGM và simulator RL đều là mô phỏng do dự án tạo; RL không vượt công thức chuẩn.
- Nguy cơ biến chứng là công thức tự đặt, không phải học máy.
- Chưa có xác thực, chỉ chạy cục bộ; chưa lưu trữ theo thời gian, chưa dùng Neo4j.
- Xem số đo và giới hạn từng phần ở `docs/metrics.md`.
