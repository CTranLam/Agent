# Hệ đa agent hỗ trợ quản lý tiểu đường

Demo môn học Phát triển các hệ thống thông minh. Là công cụ hỗ trợ học tập, **không phải thiết bị y tế, không thay thế bác sĩ**; cơ sở tri thức là bản tóm tắt giáo dục chưa được thẩm định.

## Nguồn tri thức

Khuyến nghị trong `knowledge_docs/` là bản tóm tắt ngắn bằng lời của đồ án từ các bài viết trên vinmec.com, mỗi mục có link `url`; không lưu nguyên văn bài.

Nguồn PDF: `knowledge_docs/pdf/QD_5481_BYT_2020.pdf`, Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2 của Bộ Y tế (QĐ 5481/QĐ-BYT ngày 30/12/2020), tải từ canhgiacduoc.org.vn. `kb/pdf_docs.py` trích nguyên văn 20 đoạn khai trong `knowledge_docs/pdf/sources.json` (trang, mốc đầu, mốc cuối) ra `knowledge_docs/pdf_byt_5481.md`, chỉ viết đủ chữ viết tắt (BN, ĐTĐ) và sửa lỗi lớp chữ của PDF; link nguồn mở đúng trang. QĐ 1353/QĐ-BYT (2021) chỉ sửa trang 37, không thuộc các đoạn này. Không sửa tay file `.md` đó; đổi `sources.json` rồi chạy:

```bash
.venv/bin/python -m diabetes_mas.kb.pdf_docs     # PDF -> .md, sau đó nạp Neo4j và xuất lại (xem phần Neo4j)
```

Các nguồn có thể không thống nhất với nhau (ví dụ giới hạn muối). Thực phẩm, tương tác, thuốc thay thế và các ngưỡng an toàn trong code là ước lượng hoặc theo tiêu chuẩn riêng, không lấy từ các nguồn trên. Toàn bộ chưa được bác sĩ, dược sĩ thẩm định.

## Sơ đồ kiến trúc và mã nguồn

| Khối trên sơ đồ | Hiện thực |
|---|---|
| DB có sẵn | `data/diabetes.csv`, CGM mô phỏng `dl/synthetic.py` |
| Deep learning → Phân loại | `DiabetesNet` (`dl/models.py`, `dl/train.py`) + tiêu chuẩn ADA (`dl/decision.py`) |
| Deep learning → Dự đoán | `GlucoseLSTM` (`dl/train_lstm.py`), dự báo bảo thủ |
| Web, PDF → KB | `knowledge_docs/*.md` (PDF qua `kb/pdf_docs.py`), `kb/foods.json`, `kb/drugs.json`, `kb/clinical.json` nạp vào đồ thị Neo4j (`kb/neo4j_kb.py`); dự phòng trong bộ nhớ (`kb/store.py`) |
| Knowledge Graph lâm sàng | Patient, Disease, Symptom, RiskFactor, Measurement (`kb/clinical.py`); truy vấn nhiều bước nhảy `kb/kg_queries.py`, xem `docs/cypher_examples.md` |
| RAG: truy xuất | `kb/rag.py`: BM25 + bigram + đồng nghĩa (mặc định), vector embedding cosine và lai RRF (`kb/vector.py`) |
| RAG: sinh câu trả lời | `kb/generate.py`: LLM tuỳ chọn, chỉ dựa trên đoạn trích, kiểm tra trích dẫn rồi mới hiển thị |
| Reinforcement learning | `RLAgent` (`agents/rl_agent.py`), bảng Q trong `policy/rl_insulin.py` |
| Rule | `RuleAgent` (`agents/rule_agent.py`), luật trong `policy/safety_rules.py`; `policy/exercise_gate.py`, `agents/auditor.py` |
| Agent | `Orchestrator` điều phối 5 agent: `DiagnosisAgent` (nhận kết quả DL), `GlucoseAgent`, `NutritionAgent`, `MedicationAgent`, `ExerciseAgent` |
| Tư vấn, trợ giúp | `api.py`, `web/index.html` |

## RLAgent và RuleAgent

`MedicationAgent` gọi hai agent con theo thứ tự: **`RLAgent` đề xuất → `RuleAgent` quyết định**. Chế độ khẩn cấp không chạy agent nào trong hai agent này.

| | `RLAgent` | `RuleAgent` |
|---|---|---|
| Vai trò | Chỉ đề xuất liều insulin bù | Quyết định liều cuối: giữ, giảm hoặc chặn, không bao giờ tăng |
| Đầu vào | Đường huyết, xu hướng mỗi 5 phút, ISF | Đề xuất của RL + hồ sơ, dự báo 30 phút, lần tiêm gần nhất, độ mới của số đo, carb |
| Cách quyết định | Bảng Q: chọn hành động có giá trị cao nhất trong 7 mức 0-4 đv | Luật cố định R0-R6, trần liều; liều bữa ăn = carb / ICR |
| Có học không | Có, học trước trên simulator đồ chơi (gamma = 0), không học thêm khi chạy | Không |
| Phần thưởng | Phạt khi lệch mục tiêu, phạt nặng khi dưới 70 và 54 mg/dL | Không dùng |
| Giải thích được | Hạn chế (tra bảng Q) | Có, báo mã luật đã dùng |
| Mã nguồn | `agents/rl_agent.py`, `policy/rl_insulin.py` | `agents/rule_agent.py`, `policy/safety_rules.py` |

Cách ra quyết định của từng agent chuyên môn (hiện trên thẻ kết quả): tên hiển thị trùng tên lớp trong code:

| Agent | Nhiệm vụ | Cách ra quyết định | File |
|---|---|---|---|
| `DiagnosisAgent` | Chẩn đoán | Deep learning + Rule | `agents/diagnosis_agent.py` |
| `GlucoseAgent` | Đường huyết | Deep learning + Rule | `agents/glucose_agent.py` |
| `NutritionAgent` | Dinh dưỡng | Rule + KB | `agents/nutrition_agent.py` |
| `MedicationAgent` | Thuốc & insulin | RL + Rule + KB | `agents/medication_agent.py` |
| `ExerciseAgent` | Vận động | Rule + KB | `agents/exercise_agent.py` |

## Cài và chạy

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn diabetes_mas.api:app --host 127.0.0.1 --port 8000   # mở http://127.0.0.1:8000
```

Model đã huấn luyện có sẵn trong `artifacts/`. Mặc định hệ thống không gọi API ngoài và không dùng LLM: mô hình học sâu và RL do dự án tự huấn luyện, phần hỏi đáp trả đoạn trích từ cơ sở tri thức. Bước sinh câu trả lời bằng LLM là tuỳ chọn (xem dưới).

## RAG đầy đủ: truy xuất + sinh câu trả lời (tuỳ chọn)

```
câu hỏi → bộ lọc ngoài phạm vi → truy xuất top-3 đoạn (BM25 | vector | lai)
        → [có LLM] prompt "chỉ dùng đoạn trích, mỗi câu kèm [n]" → LLM → kiểm tra → hiển thị câu sinh + đoạn trích
        → [không có LLM hoặc kiểm tra trượt] hiển thị nguyên đoạn trích kèm nguồn
```

- **Chia đoạn (chunking):** mỗi mục `## ...` trong `knowledge_docs/*.md` là một đoạn (1–3 câu, có nguồn, url, chủ đề).
- **Embedding và kho vector:** mặc định LSA (TF-IDF trên từ đơn và bigram rồi SVD còn 64 chiều, chạy offline); muốn dùng mô hình embedding thật thì đặt `EMBED_BASE_URL`, `EMBED_MODEL` (vd Ollama `nomic-embed-text`, `bge-m3`). Kho vector là ma trận numpy trong bộ nhớ, so khớp bằng cosine. Chọn bộ truy xuất bằng `RAG_RETRIEVER=bm25|vector|hybrid`; số đo so sánh ở `docs/metrics.md`.
- **Sinh câu trả lời:** đặt `LLM_BASE_URL` và `LLM_MODEL` (endpoint kiểu OpenAI `/chat/completions`). Khuyến nghị chạy LLM trên máy bằng Ollama để dữ liệu sức khoẻ không rời máy:

```bash
ollama pull qwen2.5:7b
export LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=qwen2.5:7b
.venv/bin/uvicorn diabetes_mas.api:app --host 127.0.0.1 --port 8000
```

- **Giảm ảo giác:** (1) không có đoạn trích thì không gọi LLM; (2) prompt buộc chỉ dùng đoạn trích và trích dẫn `[n]`; (3) `verify()` kiểm tra sau khi sinh: mọi câu phải có trích dẫn tới đoạn có thật, mọi con số phải có trong đoạn được trích, ít nhất 50% từ nội dung của mỗi câu phải có trong đoạn được trích. Trượt một điều là bỏ câu sinh, trả nguyên đoạn trích và ghi lý do (`fallback_reason`). Gửi `{"question": ..., "generate": false}` tới `/ask` để buộc chỉ trích đoạn.
- **Giới hạn:** kiểm tra là theo từ và số, không hiểu nghĩa; câu sinh có thể diễn đạt sai ý mà vẫn qua kiểm tra. Agent khi phân tích hồ sơ vẫn chỉ dùng phần truy xuất, không dùng câu do LLM sinh.

## Neo4j (không bắt buộc)

Không chạy Neo4j thì hệ thống tự dùng KB trong bộ nhớ. Để dùng Neo4j:

```bash
cp .env.example .env                   # rồi đổi NEO4J_PASSWORD
docker compose up -d                   # đợi khoảng 30 giây; giao diện tại http://127.0.0.1:7474
set -a; . ./.env; set +a               # nạp biến môi trường cho terminal hiện tại
.venv/bin/python -m diabetes_mas.kb.neo4j_kb   # nạp dữ liệu vào đồ thị, chạy lại khi sửa KB
.venv/bin/uvicorn diabetes_mas.api:app --host 127.0.0.1 --port 8000
```

Góc trên giao diện hiện "Tri thức neo4j" khi đang dùng đồ thị. Lệnh nạp cũng tạo phần lâm sàng: 12 bệnh, 18 triệu chứng, 10 yếu tố nguy cơ, 6 loại chỉ số và 8 bệnh nhân mẫu (lấy từ `scenarios.py`) nối tới thuốc và bệnh lý nền. Chạy thử truy vấn nhiều bước:

```bash
.venv/bin/python -m diabetes_mas.kb.kg_queries --patient "Ca 5"   # thuốc → chống chỉ định → bệnh lý nền → thuốc thay thế, ...
.venv/bin/python -m diabetes_mas.kb.kg_queries --cypher           # in các câu Cypher để chép vào Neo4j Browser
``` `KB_BACKEND=neo4j` bắt buộc dùng Neo4j, `KB_BACKEND=memory` tắt hẳn. Test tích hợp Neo4j tự bỏ qua khi chưa đặt `NEO4J_PASSWORD`.

### Xuất KB từ Neo4j ra file và ảnh

```bash
.venv/bin/python -m diabetes_mas.kb.export_kb                # đọc từ Neo4j, ghi vào exports/
.venv/bin/python -m diabetes_mas.kb.export_kb --restore exports/kb_neo4j.cypher   # dựng lại đồ thị từ file
```

Khung chat (RAG) đọc trực tiếp `exports/kb_neo4j.cypher` khi server khởi động; chưa có file thì dùng dữ liệu KB. Sau khi sửa KB, hãy nạp lại, xuất lại rồi khởi động lại server thì chat mới thấy nội dung mới.

Lệnh xuất tạo `kb_neo4j.cypher` (file dựng lại đồ thị), `kb_graph.png` và `kb_graph.svg`. Dựng lại xong cần khởi động lại server. Lệnh dựng lại xoá các nút KB hiện có rồi tạo lại, trong một giao dịch nên file lỗi thì đồ thị giữ nguyên.

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
- Chưa có xác thực, chỉ chạy cục bộ; chưa lưu trữ theo thời gian.
- Xem số đo và giới hạn từng phần ở `docs/metrics.md`.
