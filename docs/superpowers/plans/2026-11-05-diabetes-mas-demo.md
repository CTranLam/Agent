# Demo hệ đa agent hỗ trợ tiểu đường — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Web demo chạy cục bộ: nhập hồ sơ và đường huyết, nhận báo cáo tư vấn có nguồn; đi đúng sơ đồ thầy vẽ.

**Architecture:** Luồng một chiều: dữ liệu → học sâu → ngữ cảnh chung → orchestrator → 4 agent (tra KB qua RAG, liều đi qua RL rồi rule) → Auditor → báo cáo. Logic agent/rule không phụ thuộc PyTorch nên test được bằng `FakePredictor`. LLM là tuỳ chọn và chỉ diễn đạt lại, không quyết định liều.

**Tech Stack:** Python 3.12, PyTorch, numpy, pandas, scikit-learn (baseline và metric), FastAPI, pytest, HTML/JS thuần, `anthropic` (tuỳ chọn).

**Spec:** Không có file riêng; mục "Đặc tả" dưới đây là spec, rút từ ảnh bảng của thầy và báo cáo mẫu.

## Đặc tả

### Sơ đồ của thầy → hiện thực

| Khối trên bảng | Hiện thực trong demo | Task |
|---|---|---|
| DB (có sẵn) | `data/diabetes.csv` + CGM mô phỏng | 2, 4 |
| Deep learning | `DiabetesNet`, `GlucoseLSTM` | 2, 4 |
| Dự đoán | Đường huyết 30 phút tới | 4 |
| Phân loại | Không bệnh / tiền tiểu đường / T1 / T2 | 2, 3 |
| Web, book → "tạo" → KB | `knowledge_docs/*.md` + `kb/*.json` → `load_kb()` | 5 |
| RAG | BM25 + bigram + đồng nghĩa; LLM tuỳ chọn | 5, 9 |
| Reinforcement learning | Q-learning dạng bảng đề xuất liều bù | 6 |
| Rule | `apply_safety`, cổng vận động, Auditor | 6, 8 |
| Agent | Orchestrator + 4 agent chuyên trách | 7, 8 |
| Tư vấn, trợ giúp | Báo cáo tư vấn + khung hỏi đáp trên web | 10 |

### Kịch bản demo (cũng là test tích hợp, Task 8)

| # | Ca | Kết quả phải thấy |
|---|---|---|
| 1 | Nam 55t, 60kg/170cm, HbA1c 5,1, G 100, 1 số đo | Không tiểu đường; ngân sách 60g/bữa, ~1810 kcal; cảnh báo chất lượng dữ liệu; không có TIR |
| 2 | Nam 50t, 51kg/170cm, HbA1c 5,8, G 124 | Tiền tiểu đường |
| 3 | Nam 30t, 55kg/163cm, HbA1c 6,7, G 113, xác suất DL dưới ngưỡng | T2, nguồn ghi "tiêu chuẩn ADA" |
| 4 | T1, ISF 40, ICR 12, CGM 205→215, carb 60, tiêm 3 đv cách 90 phút | Bù 0,5 đv (RL giả 2,0 − IOB 1,5), bữa ăn 5,0 đv |
| 5 | 80t, gliclazide, eGFR 25, G 48 | Khẩn cấp; không liều; không thực đơn/vận động; vẫn cảnh báo gliclazide |
| 6 | Metformin, G 420 | Khẩn cấp, cảnh báo critical, không liều |
| 7 | Suy tim + pioglitazone | Chống chỉ định + gợi ý thuốc thay thế |
| 8 | T1, G 265 | Không được tập; không có câu "đi bộ" ở bất kỳ agent nào |

### Không làm trong bản này (so với báo cáo mẫu)

Neo4j và tab đồ thị (đồ thị đặt trong JSON), SQLite/nhật ký/nhắc thuốc/vòng phản hồi, embedding rerank, API key/rate limit (chỉ chạy `127.0.0.1`), kiểm thử đột biến. Nếu còn thời gian thì làm theo thứ tự: tab đồ thị → SQLite → kiểm thử đột biến.

## Global Constraints

- Số đo ở mg/dL. Nhận 20–600; nếu mọi giá trị ≤ 35 thì `ValueError` chứa "mmol/L".
- `config.py`: `HYPO_L1=70, HYPO_L2=54, HYPER_KETONE=250, HYPER_SEVERE=300, HYPER_CRITICAL=400, TARGET_GLUCOSE=110, MAX_CORRECTION_DOSE=4.0, MAX_MEAL_BOLUS=10.0, MAX_TOTAL_BOLUS=12.0, INSULIN_ACTION_MIN=180, READING_STALE_MIN=20, MIN_READINGS_FOR_DOSE=4, MIN_READINGS_FOR_STATS=12, CARB_SHARE=0.5, SNACK_CARB_G=15`.
- Liều làm tròn **xuống** bội 0,5 (`floor(x*2)/2`) để không bao giờ vượt trần.
- Không biết thì không tiêm: thiếu dữ liệu luôn cho liều 0, không đoán.
- Mọi số liệu đưa vào báo cáo phải do chính code này đo, không chép từ báo cáo mẫu; kết quả trên dữ liệu mô phỏng phải ghi rõ là mô phỏng.
- Mọi báo cáo có câu miễn trừ: "Công cụ hỗ trợ học tập, không phải thiết bị y tế, không thay thế bác sĩ."
- KB là bản tóm tắt giáo dục, `reviewed: false`, nguồn ghi đúng tên tài liệu (ADA Standards of Care, WHO, Bộ Y tế), không tự tạo số trang hay trích dẫn.
- Giao diện và thông điệp tiếng Việt; tên trong code tiếng Anh.
- Commit message kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Người dùng nhập đơn vị mmol/L (ví dụ 5,6): báo lỗi rõ ràng, không coi là hạ đường huyết nặng. Test ở Task 8.
2. Chỉ có 1 số đo: dự báo `low_confidence`, không TIR, không liều bù. Test ở Task 4, 6, 7.
3. Thuốc gõ tự do hoặc không có trong KB ("Metformin 850mg", "thuoc la"): nhận ra được tên có liều; tên lạ phải báo "cần dược sĩ kiểm tra", không bỏ qua im lặng. Test ở Task 7.
4. Câu hỏi rỗng hoặc ngoài phạm vi ở khung hỏi đáp: trả "không tìm thấy trong cơ sở tri thức", không trả đoạn ngẫu nhiên. Test ở Task 5.
5. Giá trị vô lý ở API (cân nặng 0, ISF 0, số đo NaN, danh sách rỗng): trả 422, không phải 500. Test ở Task 10.

---

## Cấu trúc file

```
requirements.txt  .gitignore  README.md
data/diabetes.csv                 # người dùng tải về, không commit
artifacts/                        # model đã huấn luyện, meta.json, q_table.npy
knowledge_docs/*.md               # "web, sách" → KB
docs/metrics.md                   # sinh bởi eval.report_metrics
diabetes_mas/
  config.py schemas.py scenarios.py llm.py guardrail.py api.py web/index.html
  dl/      models.py features.py train.py synthetic.py train_lstm.py decision.py predictor.py
  kb/      foods.json drugs.json store.py rag.py
  policy/  rl_insulin.py safety_rules.py exercise_gate.py
  agents/  base.py conditions.py glucose_agent.py nutrition_agent.py medication_agent.py
           exercise_agent.py auditor.py orchestrator.py
  eval/    rag_eval.py rag_gold.json report_metrics.py
tests/     fakes.py test_*.py
```

---

### Task 1: Khung dự án và kiểu dữ liệu chung

**Files:**
- Create: `requirements.txt`, `.gitignore`, `diabetes_mas/__init__.py`, `diabetes_mas/config.py`, `diabetes_mas/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces, trong `schemas.py` (dataclass):
  - `PatientProfile(name="", age=50, sex="M", weight_kg=70.0, height_cm=165.0, hba1c=5.5, fasting_glucose: float|None=None, duration_years=0, hypertension=False, heart_disease=False, heart_failure=False, pregnant=False, mtc_men2_history=False, pancreatitis_history=False, smoking="never", egfr=90.0, c_peptide: float|None=None, autoantibody: bool|None=None, on_insulin=False, medications: list[str]=[], isf=40.0, icr=12.0, diabetes_type_known: str|None=None, allergies: list[str]=[], vegetarian=False)`, property `bmi`
  - `Evidence(source, title, text, score=0.0)`
  - `Alert(level, message, agent)`, với `level` thuộc `"critical"|"warning"|"info"`
  - `AgentResult(agent, summary, details: dict, actions: list[str], alerts: list[Alert], evidence: list[Evidence])`
  - `AnalysisContext(profile, readings: list[float], classification: dict, forecast: dict, planned_carbs: float|None=None, minutes_since_insulin: float|None=None, last_insulin_units: float|None=None, reading_age_min: float|None=None, meal_type: str|None=None, shared: dict={})`, properties `glucose` (số đo cuối), `n_readings`, `forecast_min` (min của `forecast["values_conservative"]`), `diabetes_type` (`classification["code"]`)
  - `Report(plan, steps, classification, forecast, results: dict[str,AgentResult], alerts, audit_findings: list[str], synthesis: str|None, synthesis_note: str|None, sources: list[str], disclaimer)`, có `to_dict()` và `to_text()`

- [ ] **Step 1: Tạo môi trường**

Run: `git init && python3 -m venv .venv && .venv/bin/pip install numpy pandas torch scikit-learn fastapi uvicorn httpx pytest anthropic`, rồi ghi phiên bản đã cài vào `requirements.txt`. `.gitignore` gồm `.venv/`, `data/`, `__pycache__/`, `.env`.
Expected: `.venv/bin/python -c "import torch, fastapi; print(torch.cuda.is_available())"` chạy không lỗi.

- [ ] **Step 2: Viết test thất bại**

```python
def test_bmi():
    assert round(PatientProfile(weight_kg=70, height_cm=165).bmi, 1) == 25.7

def test_defaults_not_shared():
    a, b = PatientProfile(), PatientProfile()
    a.medications.append("x")
    assert b.medications == []

def test_unknown_labs_stay_none():
    p = PatientProfile()
    assert p.c_peptide is None and p.autoantibody is None and p.fasting_glucose is None

def test_context_shortcuts():
    ctx = AnalysisContext(PatientProfile(), [100.0, 90.0], {"code": "T2"},
                          {"values_conservative": [80.0, 75.0]})
    assert (ctx.glucose, ctx.n_readings, ctx.forecast_min, ctx.diabetes_type) == (90.0, 2, 75.0, "T2")

def test_report_to_dict_is_json_safe():
    json.dumps(Report("thường quy", [], {}, {}, {}, [], [], None, None, [], "d").to_dict())
```

- [ ] **Step 3: Chạy test**

Run: `.venv/bin/pytest tests/test_schemas.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Cài đặt `config.py` (đúng giá trị ở Global Constraints) và `schemas.py`**

- [ ] **Step 5: Chạy test**

Run: `.venv/bin/pytest tests/test_schemas.py -v`
Expected: PASS.

- [ ] **Step 6: Commit** — `feat: scaffold project and shared schemas`

---

### Task 2: DiabetesNet (Phân loại)

**Cần người dùng:** tải "Diabetes prediction dataset" trên Kaggle, đặt tại `data/diabetes.csv` (cột `gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes`; kỳ vọng 100.000 dòng, ~8,5% dương). Test không cần file này.

**Files:**
- Create: `diabetes_mas/dl/models.py`, `diabetes_mas/dl/features.py`, `diabetes_mas/dl/train.py`
- Test: `tests/test_dl_classifier.py`

**Interfaces:**
- Produces:
  - `FEATURES: list[str]`, 13 cột: `age, hypertension, heart_disease, bmi, hba1c, glucose, gender_Female, gender_Male, smoking_never, smoking_former, smoking_current, smoking_ever, smoking_not_current`. "No Info" và gender "Other" là toàn 0.
  - `encode_df(df) -> np.ndarray`, `encode_profile(p: PatientProfile, glucose: float) -> np.ndarray` (trả về shape `(13,)`)
  - `DiabetesNet(n_features, hidden=(64,32,16,8), dropout=0.1)`, trả về logit shape `(B,)`
  - `split_dedup(df, seed=42) -> (train, val, test)`: bỏ dòng trùng rồi chia 70/15/15 phân tầng
  - `pick_threshold(y_true, probs, min_recall=0.90) -> float`: ngưỡng cao nhất vẫn đạt recall
  - `train(csv="data/diabetes.csv", out="artifacts")`: ghi `diabetes_net.pt` và `meta.json` gồm `mean, std, threshold, metrics, baselines`

- [ ] **Step 1: Viết test thất bại**

```python
def fake_df(n=400): ...  # hba1c >= 6.5 thì diabetes=1, đủ các giá trị gender/smoking

def test_feature_count_and_alignment():
    assert encode_df(fake_df()).shape[1] == 13 == len(FEATURES)
    row = pd.DataFrame([{"gender": "Male", "age": 30, "hypertension": 0, "heart_disease": 0,
                         "smoking_history": "never", "bmi": 25.0, "HbA1c_level": 6.0,
                         "blood_glucose_level": 110, "diabetes": 0}])
    p = PatientProfile(age=30, sex="M", weight_kg=25.0, height_cm=100, hba1c=6.0, smoking="never")
    assert np.allclose(encode_df(row)[0], encode_profile(p, 110.0))

def test_no_info_is_all_zero_smoking():
    i = [k for k, f in enumerate(FEATURES) if f.startswith("smoking_")]
    assert encode_df(fake_df().assign(smoking_history="No Info"))[:, i].sum() == 0

def test_output_is_logit_shape():
    assert DiabetesNet(13)(torch.zeros(5, 13)).shape == (5,)

def test_split_has_no_row_overlap():
    tr, va, te = split_dedup(pd.concat([fake_df(), fake_df()]))
    assert not set(map(tuple, tr.values)) & set(map(tuple, te.values))

def test_threshold_reaches_recall():
    y, p = np.array([1, 1, 1, 0, 0]), np.array([.9, .8, .2, .5, .1])
    t = pick_threshold(y, p, 0.9)
    assert ((p >= t)[y == 1]).mean() >= 0.9 and t == 0.2
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_dl_classifier.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt `features.py`, `models.py`, `train.py`**

Adam, `BCEWithLogitsLoss`, dừng sớm theo val loss (patience 5, tối đa 50 epoch). Mean/std chỉ tính trên train và chỉ áp dụng cho 6 cột số. `meta.json` lưu thêm accuracy, precision, recall, F1, ROC-AUC, PR-AUC trên test, cùng baseline: chỉ HbA1c, chỉ glucose, logistic 13 đặc trưng.

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_dl_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Huấn luyện thật**

Run: `.venv/bin/python -m diabetes_mas.dl.train`
Expected: có `artifacts/meta.json`, recall trên test ≥ ~0,88. Ghi số vào báo cáo cùng baseline; nhãn dataset gần như hai ngưỡng nên ROC-AUC cao không có nghĩa dự báo tốt trên người thật.

- [ ] **Step 6: Commit** — `feat: DiabetesNet training with leak-free split and baselines`

---

### Task 3: Từ xác suất đến loại bệnh

**Files:**
- Create: `diabetes_mas/dl/decision.py` (không import torch)
- Test: `tests/test_decision.py`

**Interfaces:**
- Consumes: `PatientProfile`
- Produces:
  - `meets_ada_diabetes(p) -> bool`: HbA1c ≥ 6,5 hoặc `fasting_glucose` nhập ≥ 126. Giá trị đói ước lượng không tính, nên chỉ dùng `p.fasting_glucose` do người dùng nhập.
  - `is_prediabetes(p) -> bool`: HbA1c 5,7–6,4 hoặc đói 100–125
  - `infer_type(p) -> "T1"|"T2"`: `score = 2*(autoantibody is True) + 2*(c_peptide is not None and c_peptide < 0.6) + 1*(max(age-duration,0) < 30)`; `T1` nếu `score >= 3`
  - `complication_risks(p) -> dict`: `retinopathy`, `nephropathy`, `neuropathy` (sigmoid của `z`) và `"label": "ước lượng theo quy tắc, không phải mô hình học máy"`
    - retinopathy `z = -8 + 0.6*hba1c + 0.1*duration + 0.01*age`
    - nephropathy `z = -7.5 + 0.5*hba1c + 0.08*duration + 0.02*max(0,60-egfr) + 0.5*hypertension`
    - neuropathy `z = -8 + 0.5*hba1c + 0.1*duration + 0.02*age`
  - `classify_with_prob(p, prob, thr) -> dict`: khoá `code ("NONE"|"PRE"|"T1"|"T2"), class_name, confidence, source, risk_level, type_low_confidence, complications`. Thứ tự: type đã chẩn đoán → `prob>=thr` → ADA → tiền tiểu đường → không bệnh.
  - `conservative_forecast(window: list[float], lstm_vals: list[float], n_input: int, in_len=12) -> dict`: khoá `values, values_conservative, minutes, trend_per_5min, n_input, low_confidence`. Slope lấy từ 4 điểm cuối của `window`; `values_conservative = clip(min(lstm, linear), 30, 450)`.

- [ ] **Step 1: Viết test thất bại**

```python
def test_known_type_wins():
    assert classify_with_prob(PatientProfile(diabetes_type_known="T1"), .0, .09)["code"] == "T1"

def test_ada_net_catches_low_prob():
    r = classify_with_prob(PatientProfile(age=30, hba1c=6.7), .01, .09)
    assert r["code"] == "T2" and "ADA" in r["source"]

def test_prediabetes_and_none():
    assert classify_with_prob(PatientProfile(hba1c=5.8), .01, .09)["code"] == "PRE"
    assert classify_with_prob(PatientProfile(hba1c=5.1), .01, .09)["code"] == "NONE"

def test_missing_labs_score_zero():
    assert infer_type(PatientProfile(age=20)) == "T2"
    assert classify_with_prob(PatientProfile(age=20, hba1c=8), .9, .09)["type_low_confidence"] is True

def test_t1_inference():
    assert infer_type(PatientProfile(age=20, autoantibody=True, c_peptide=.1)) == "T1"

def test_complications_labelled_and_monotonic():
    a = complication_risks(PatientProfile(hba1c=6)); b = complication_risks(PatientProfile(hba1c=10))
    assert b["retinopathy"] > a["retinopathy"] and "quy tắc" in a["label"]

def test_conservative_takes_min_and_flags():
    f = conservative_forecast([100, 95, 90, 85], [200.0]*6, 4)
    assert f["values_conservative"][0] == 80 and f["trend_per_5min"] == -5
    assert f["low_confidence"] is True and f["minutes"][-1] == 30
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_decision.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt `decision.py`**

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_decision.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat: rule-backed classification and conservative forecast`

---

### Task 4: GlucoseLSTM (Dự đoán) và Predictor

**Files:**
- Create: `diabetes_mas/dl/synthetic.py`, `diabetes_mas/dl/train_lstm.py`, `diabetes_mas/dl/predictor.py`, `tests/fakes.py`
- Modify: `diabetes_mas/dl/models.py` (thêm `GlucoseLSTM`)
- Test: `tests/test_forecast.py`

**Interfaces:**
- Consumes: `conservative_forecast`, `classify_with_prob`, `encode_profile`
- Produces:
  - `GlucoseLSTM(hidden=64, horizon=6)`, input `(B,T,1)`, output `(B,6)`
  - `simulate_cgm(n_series: int, length=288, seed=0) -> np.ndarray`: mô phỏng bữa ăn, liều và nhiễu; phải có đoạn < 70
  - `split_by_series(series, seed=0) -> (train, val, test)` và `windows(series, in_len=12, out_len=6) -> (X, Y)`: chia theo chuỗi trước, cắt cửa sổ sau, tránh rò rỉ
  - `forecast_metrics(y_true, y_pred) -> {"mae", "hypo_recall"}`: `hypo_recall` là tỉ lệ cửa sổ có thực tế < 70 trong 30 phút tới và dự báo cũng < 70
  - `train_lstm(out="artifacts")`: so LSTM với "giữ nguyên" và "tuyến tính", ghi `lstm.pt`, thêm `forecast_metrics` vào `meta.json`
  - `Predictor(artifacts="artifacts")`, `.classify(p) -> dict`, `.forecast(readings) -> dict`. Thiếu artifact thì `RuntimeError` nêu lệnh cần chạy. Đệm cửa sổ bằng giá trị đầu khi có < 12 số đo.
  - `tests/fakes.py`: `FakePredictor(prob=0.01, thr=0.09)` cùng giao diện, dùng `classify_with_prob` và `conservative_forecast` thật với `lstm_vals` là ngoại suy tuyến tính

- [ ] **Step 1: Viết test thất bại**

```python
def test_split_by_series_no_overlap():
    tr, va, te = split_by_series(simulate_cgm(20, seed=1))
    assert not {a.tobytes() for a in tr} & {a.tobytes() for a in te}

def test_windows_shape():
    X, Y = windows(simulate_cgm(2, length=30, seed=1))
    assert X.shape[1:] == (12, 1) and Y.shape[1:] == (6,)

def test_simulator_has_hypo_events():
    assert (simulate_cgm(20, seed=1) < 70).any()

def test_metrics_hold_last_misses_falling_hypo():
    y = np.array([[80, 75, 70, 65, 60, 55.]]); hold = np.full((1, 6), 85.)
    assert forecast_metrics(y, hold)["hypo_recall"] == 0.0
    assert forecast_metrics(y, y)["hypo_recall"] == 1.0

def test_single_reading_low_confidence():  # Review Focus 2
    f = FakePredictor().forecast([120.0])
    assert f["low_confidence"] and f["trend_per_5min"] == 0 and f["n_input"] == 1

def test_predictor_missing_artifact_message(tmp_path):
    with pytest.raises(RuntimeError, match="train"):
        Predictor(tmp_path)
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_forecast.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt các file trên**

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_forecast.py -v`
Expected: PASS.

- [ ] **Step 5: Huấn luyện thật**

Run: `.venv/bin/python -m diabetes_mas.dl.train_lstm`
Expected: `meta.json` có `mae` và `hypo_recall` của cả 3 phương pháp. Nếu LSTM không hơn baseline thì vẫn báo cáo đúng như vậy.

- [ ] **Step 6: Commit** — `feat: GlucoseLSTM, CGM simulator, predictor facade`

---

### Task 5: Web/sách → KB → RAG (truy xuất)

**Files:**
- Create: `knowledge_docs/*.md`, `diabetes_mas/kb/foods.json`, `diabetes_mas/kb/drugs.json`, `diabetes_mas/kb/store.py`, `diabetes_mas/kb/rag.py`, `diabetes_mas/eval/rag_gold.json`, `diabetes_mas/eval/rag_eval.py`
- Test: `tests/test_kb_rag.py`

**Interfaces:**
- Produces:
  - Định dạng tài liệu: mỗi khuyến nghị là `## tiêu đề`, tiếp theo dòng `source: ...`, `topic: ...`, rồi nội dung. Tối thiểu 25 khuyến nghị, phủ các chủ đề `hypoglycemia, hyperglycemia, nutrition, exercise, medication, insulin, complications, special_groups`.
  - `foods.json`: tối thiểu 25 món, có món Việt, trường `name, category(starch|protein|vegetable|fruit|fast_carb), carb_g, portion, gi, meals, allergens, high_sodium, vegetarian`.
  - `drugs.json`: khoá `drugs` (tối thiểu metformin, gliclazide, pioglitazone, sitagliptin, linagliptin, empagliflozin, liraglutide, insulin aspart, insulin glargine; trường `name, aliases, contraindications, cautions, not_for_types`), `interactions` (`a, b, severity, message`), `substitutes` (`drug, condition, alternative, why`). Tối thiểu: metformin chống chỉ định `egfr_lt_30`; pioglitazone chống chỉ định `heart_failure`; gliclazide thận trọng `egfr_lt_30`, `age_ge_75`; liraglutide chống chỉ định `mtc_men2`.
  - `KnowledgeBase`, tạo bằng `load_kb() -> KnowledgeBase`: `.recommendations`, `.foods`, `.drugs`, `.match_drug(text) -> dict|None`, `.interactions(names)`, `.alternatives(drug, condition) -> list[dict]`, `.meta == {"reviewed": False, ...}`
  - `fold(s)`, `expand_query(q)`, `features(s)` (token + bigram), `RAG(kb)`, `.retrieve(query, k=3) -> list[Evidence]`, `.extractive(evidence) -> str`. Tiêu đề tính trọng số gấp đôi. Điểm = BM25 chuẩn hoá về 0–1 (chia cho điểm cao nhất) + 0,5 nếu đoạn khớp chủ đề suy từ query. BM25 thô bằng 0 thì trả `[]`.

- [ ] **Step 1: Viết test thất bại**

```python
def test_slang_finds_hypo():
    assert "hạ đường huyết" in RAG(load_kb()).retrieve("bị tụt đường phải làm gì")[0].title.lower()

def test_every_evidence_has_source():
    assert all(e.source for e in RAG(load_kb()).retrieve("ăn gì tốt", k=5))

def test_empty_or_off_topic_returns_nothing():  # Review Focus 4
    r = RAG(load_kb())
    assert r.retrieve("") == [] and r.retrieve("   ") == [] and r.retrieve("xyzzy qwerty") == []
    assert "không tìm thấy" in r.extractive([]).lower()

def test_match_drug_with_dose():  # Review Focus 3
    kb = load_kb()
    assert kb.match_drug("Metformin 850mg")["name"].lower() == "metformin"
    assert kb.match_drug("thuoc la") is None

def test_alternatives():
    assert load_kb().alternatives("pioglitazone", "heart_failure")

def test_kb_not_marked_reviewed():
    assert load_kb().meta["reviewed"] is False
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_kb_rag.py -v`
Expected: FAIL.

- [ ] **Step 3: Viết KB và `store.py`, `rag.py`**

Nội dung KB tóm tắt từ ADA Standards of Care, WHO và Bộ Y tế; đối chiếu nguồn thật khi viết, không viết từ trí nhớ rồi gán nguồn.

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_kb_rag.py -v`
Expected: PASS.

- [ ] **Step 5: Đánh giá truy xuất**

Viết `rag_gold.json` (≥ 20 câu hỏi đời thường, kèm tiêu đề đoạn đúng) trước khi chỉnh đồng nghĩa. `rag_eval.py` so sánh đếm từ trùng với BM25.
Run: `.venv/bin/python -m diabetes_mas.eval.rag_eval`
Expected: in Recall@3 và MRR của cả hai. Ghi rõ trong báo cáo rằng bộ câu hỏi do chính người viết KB soạn.

- [ ] **Step 6: Commit** — `feat: knowledge base from docs and BM25 RAG`

---

### Task 6: RL + Rule an toàn

**Files:**
- Create: `diabetes_mas/policy/rl_insulin.py`, `diabetes_mas/policy/safety_rules.py`, `diabetes_mas/policy/exercise_gate.py`
- Test: `tests/test_policy.py`, `tests/test_safety_properties.py`

**Interfaces:**
- Consumes: `PatientProfile`, `AnalysisContext`, `config`
- Produces:
  - `ACTIONS = [0, .5, 1, 1.5, 2, 3, 4]`
  - `encode(g, trend, isf) -> tuple`: `digitize` với glucose `np.arange(70,310,15)` (17 ô), trend `[-6,-3,-1,1,3,6]` (7 ô), ISF `[20,25,30,35,40,50,60,80]` (9 ô)
  - `reward_fn(g1)`: `-abs(g1-110)/20`; thêm −10 nếu < 70, thêm −20 nếu < 54, thêm −1 nếu > 180
  - `Simulator(seed)`, `.reset()` (G ~ U(60,350), trend ~ U(−6,6), ISF ~ U(20,80)), `.step(units)`: `g1 = g + 6*trend - isf*units*eff + N(0,8) + N(0,5)`, `eff ~ U(.85,1.15)`
  - `QPolicy`, `.train(steps, seed)`, `.save(path)`, `.load(path)`, `.suggest(glucose, trend5, isf) -> float`; γ = 0
  - `evaluate(policy_fn, n=20000, seed=1) -> {"tir", "dose_hypo", "above_180"}`: `dose_hypo` chỉ đếm ca cuối < 70 mà không tiêm thì ≥ 70, dùng cùng nhiễu
  - CLI `python -m diabetes_mas.policy.rl_insulin [--eval]`
  - `iob_fraction(minutes, dia=180)`: `1-(3x²-2x³)`, `x=minutes/dia`, chặn trong 0–1
  - `apply_safety(profile, glucose, forecast_min, proposed, minutes_since_insulin, dtype, *, n_readings=None, reading_age_min=None, last_insulin_units=None) -> SafetyDecision(dose, proposed, iob, blocked, rules_fired: list[str], alerts: list[Alert])`. Thứ tự: R1 < 54; R2 < 70; R3 dự báo < 70; R4 không dùng insulin; R0a số đo > 20 phút; R0b < 4 số đo; R5 trừ IOB, không rõ liều trước thì chặn; R6 trần `min(dose, 4, max(0,(g-110)/isf - iob))`. Liều > 0 phải kèm alert bắt đầu bằng `"Liều chỉ mang tính THAM KHẢO"`.
  - `check_meal_bolus(profile, glucose, forecast_min, carbs, *, reading_age_min=None) -> (dose, notes)`
  - `cap_total_bolus(correction, meal) -> (correction, meal, notes)`: giảm liều bù trước
  - `hyper_alerts(glucose, on_insulin, dtype) -> list[Alert]`: ≥ 250 (insulin/T1) nhắc ceton; ≥ 300 liên hệ nhóm điều trị trong ngày; ≥ 400 critical; không phụ thuộc thuốc
  - `exercise_gate(ctx) -> ExerciseGate(allowed, intensity, reason)`: < 70 hoặc dự báo < 70 → không; ≥ 300 → không; > 250 và (insulin hoặc T1) → không; < 90 → cho phép mức nhẹ, ăn nhẹ 15 g carb trước; suy tim hoặc ≥ 75 tuổi → nhẹ; còn lại → trung bình

- [ ] **Step 1: Viết test thất bại**

```python
T1 = lambda **k: PatientProfile(on_insulin=True, isf=40, icr=12, **k)

def test_iob_example_from_spec():
    assert iob_fraction(90) == 0.5
    d = apply_safety(T1(), 220, 200, 2.0, 90, "T1", n_readings=12, reading_age_min=0, last_insulin_units=3)
    assert d.dose == 0.5 and d.iob == 1.5

@pytest.mark.parametrize("kw", [dict(g=50), dict(g=65), dict(fmin=60), dict(n=2), dict(age=30), dict(mins=60, last=None)])
def test_rules_force_zero(kw): ...  # dựng từ baseline g=220, fmin=200, n=12, age=0, mins=None; dose phải bằng 0

def test_not_on_insulin_zero():
    assert apply_safety(PatientProfile(), 300, 280, 3.0, None, "T2", n_readings=12).dose == 0

def test_meal_requires_user_carbs():
    assert check_meal_bolus(T1(), 150, 140, None)[0] == 0
    assert check_meal_bolus(T1(), 150, 140, 60)[0] == 5.0
    assert check_meal_bolus(T1(), 60, 60, 60)[0] == 0

def test_total_cap_cuts_correction_first():
    assert cap_total_bolus(4.0, 10.0)[:2] == (2.0, 10.0)

def test_hyper_alerts_without_insulin():
    assert any(a.level == "critical" for a in hyper_alerts(420, False, "T2"))

def test_exercise_gate_t1_265():
    assert not exercise_gate(ctx_for(T1(), 265)).allowed

def test_rl_close_to_trend_formula():
    q = QPolicy(); q.train(300_000, seed=0)
    rl, none, formula = evaluate(q.suggest), evaluate(lambda *a: 0.0), evaluate(trend_formula)
    assert rl["tir"] > none["tir"] + 0.2 and rl["tir"] > formula["tir"] - 0.07
```

Trong `tests/test_safety_properties.py`, chạy 5.000 tổ hợp `apply_safety` và 3.000 tổ hợp `check_meal_bolus` + `cap_total_bolus`, `default_rng(42)`, kiểm tra bất biến:

- `0 <= dose <= min(proposed, 4)`
- `g<70` hoặc `fmin<70` → 0; không dùng insulin → 0; `n<4` → 0; `age>20` → 0
- tiêm gần đây mà không rõ liều và `proposed>0` → 0
- `dose>0` → `dose <= max(0,(g-110)/isf)` và có alert "THAM KHẢO"
- tổng liều ≤ 12

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_policy.py tests/test_safety_properties.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt ba module**

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_policy.py tests/test_safety_properties.py -v`
Expected: PASS.

- [ ] **Step 5: Huấn luyện và đánh giá RL**

Run: `.venv/bin/python -m diabetes_mas.policy.rl_insulin && .venv/bin/python -m diabetes_mas.policy.rl_insulin --eval`
Expected: có `artifacts/q_table.npy`, in bảng 4 chính sách (không tiêm, công thức, công thức có xu hướng, RL). Ghi đúng kết quả; không được nói RL "tốt hơn" nếu không đo được thế.

- [ ] **Step 6: Commit** — `feat: Q-learning dose suggester behind safety rules`

---

### Task 7: Bốn agent chuyên trách

**Files:**
- Create: `diabetes_mas/agents/base.py`, `conditions.py`, `glucose_agent.py`, `nutrition_agent.py`, `medication_agent.py`, `exercise_agent.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `KnowledgeBase`, `RAG`, `AnalysisContext`, `AgentResult`, `QPolicy.suggest`, `apply_safety`, `check_meal_bolus`, `cap_total_bolus`, `hyper_alerts`, `exercise_gate`
- Produces:
  - `BaseAgent(kb, rag)`, `.run(ctx) -> AgentResult`, class attribute `name`
  - `conditions.evaluate(code, profile, dtype) -> bool|None`: bảng `EVALUATORS` với `egfr_lt_20/30/45/60, heart_failure, pregnancy, mtc_men2, pancreatitis_history, age_ge_75, type1_dka`; mã lạ trả `None`
  - `GlucoseAgent`: `details` có `current, mean, sd, cv_pct, tir_pct, tbr_pct, tar_pct` (4 chỉ số sau là `None` nếu < 12 số đo); cảnh báo số đo cũ > 20 phút, biến thiên > 30 mg/dL/5 phút (nghi lỗi cảm biến); hạ đường huyết → quy tắc 15 g, món từ KB loại `fast_carb`
  - `energy_kcal(p, dtype) -> (kcal, note)`: Mifflin-St Jeor × 1,3, làm tròn đến 10; BMI ≥ 23, không T1, không ≥ 75 tuổi, không BMI < 18,5, không thai → giảm 500 kcal, sàn 1500 (nam)/1200 (nữ)
  - `carb_per_meal(p, dtype) -> (grams, info)`: `(kcal*0.5/4 - 15)/3`, làm tròn đến bội 5, kẹp `[lo, 60]`; `lo=45` nếu ≥ 75 tuổi, BMI < 18,5 hoặc thai, ngược lại 30
  - `NutritionAgent`: `details["menu"]` có `items: [{name, portion, carb_g}]` và `total_carb`, xoay theo `seed = hash ổn định(profile+ngày)` (dùng `zlib.crc32`, không dùng `hash()` vì đổi theo tiến trình), lọc dị ứng, ăn chay, món mặn khi tăng huyết áp/suy tim/eGFR < 60. Chỉ khuyên đi bộ khi `exercise_gate(ctx).allowed`. Không tính liều.
  - `MedicationAgent(kb, rag, policy)`: rà soát thuốc qua `match_drug`, `evaluate`, `interactions`, `alternatives`; `details` có `correction, meal, rules_fired, unknown_drugs`; `shared["review_only"]` → `correction=meal=None`. Chỉ lấy `ctx.planned_carbs` do người dùng nhập.
  - `ExerciseAgent`: dùng `exercise_gate`; `details` có `allowed, intensity`; không cho phép thì không có khuyến nghị tập.

- [ ] **Step 1: Viết test thất bại**

```python
def test_kcal_examples_from_spec():
    p = PatientProfile(age=50, sex="M", weight_kg=70, height_cm=165)
    assert energy_kcal(p, "T2")[0] == 1430 and carb_per_meal(p, "T2")[0] == 55
    q = PatientProfile(age=55, sex="M", weight_kg=60, height_cm=170)
    assert energy_kcal(q, "NONE")[0] == 1810 and carb_per_meal(q, "NONE")[0] == 60

def test_no_tir_with_one_reading():  # Review Focus 2
    assert GlucoseAgent(kb, rag).run(ctx(readings=[100.])).details["tir_pct"] is None

def test_unknown_condition_is_none_not_false():
    assert evaluate("khong_co", PatientProfile(), "T2") is None

def test_every_kb_condition_has_evaluator():
    for d in load_kb().drugs:
        for c in d["contraindications"] + d["cautions"]:
            assert evaluate(c, PatientProfile(), "T2") is not None, c

def test_unknown_drug_reported():  # Review Focus 3
    r = med_agent().run(ctx(PatientProfile(medications=["Metformin 850mg", "thuoc la"])))
    assert r.details["unknown_drugs"] == ["thuoc la"]
    assert any("dược sĩ" in a.message for a in r.alerts)

def test_contraindication_and_alternative():
    r = med_agent().run(ctx(PatientProfile(medications=["pioglitazone"], heart_failure=True), code="T2"))
    assert any(a.level == "critical" for a in r.alerts)
    assert any("sitagliptin" in s.lower() or "linagliptin" in s.lower() for s in r.actions)

def test_no_meal_dose_without_user_carbs():
    assert med_agent().run(ctx(T1(), readings=flat(200), carbs=None)).details["meal"] == 0

def test_menu_respects_filters_and_is_deterministic():
    p = PatientProfile(vegetarian=True, allergies=["tôm"], hypertension=True)
    a, b = (NutritionAgent(kb, rag).run(ctx(p)) for _ in range(2))
    assert a.details["menu"] == b.details["menu"]
    # không có món không chay, món chứa tôm, món high_sodium

def test_no_walk_advice_when_gate_closed():
    r = NutritionAgent(kb, rag).run(ctx(T1(), readings=flat(265), code="T1"))
    assert not any("đi bộ" in a.lower() for a in r.actions)
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_agents.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt các agent**

Mỗi agent gọi `rag.retrieve` theo chủ đề của mình và gắn `evidence`.

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_agents.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat: glucose, nutrition, medication and exercise agents`

---

### Task 8: Orchestrator, Auditor và kịch bản demo

**Files:**
- Create: `diabetes_mas/agents/orchestrator.py`, `diabetes_mas/agents/auditor.py`, `diabetes_mas/scenarios.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: mọi agent, `Predictor` hoặc `FakePredictor`, `AnalysisContext`, `Report`
- Produces:
  - `validate_readings(readings) -> None`: danh sách rỗng, NaN hoặc ngoài 20–600 → `ValueError`; mọi giá trị ≤ 35 → `ValueError` chứa "mmol/L". Kiểm tra mmol/L chạy trước kiểm tra khoảng.
  - `Orchestrator(predictor, kb, rag, policy, llm=None)`
  - `.plan(ctx) -> (name, steps)`: G < 54 hoặc ≥ 400 → `"KHẨN CẤP ..."`, `shared["review_only"]=True`, chạy `["glucose"]` + `["medication"]` nếu có thuốc; ngược lại `["glucose","nutrition"]`, thêm `"medication"` nếu có thuốc/insulin, cuối cùng `"exercise"`
  - `.analyze(profile, readings, planned_carbs=None, minutes_since_insulin=None, last_insulin_units=None, reading_age_min=None, meal_type=None) -> Report`: cảnh báo gộp, sắp theo mức critical → warning → info
  - `audit(ctx, results) -> list[Finding(rule, message)]`, sửa trực tiếp `results`. A1 không liều khi đang/sắp hạ; A2 không liều bữa ăn nếu không nhập carb; A3 không liều bù nếu số đo cũ/ít; A4 trần 4 và tổng 12; A5 gỡ câu vận động khi cổng đóng; A6 G < 54 hoặc ≥ 400 luôn có critical; A7 không liều ở chế độ khẩn cấp.
  - `SCENARIOS: dict[str, dict]`: 8 ca ở mục Đặc tả, khoá `title, profile, readings, planned_carbs, minutes_since_insulin, last_insulin_units, prob` (`prob` là xác suất cho `FakePredictor`)

- [ ] **Step 1: Viết test thất bại**

```python
def test_mmol_rejected():  # Review Focus 1
    with pytest.raises(ValueError, match="mmol/L"):
        orch().analyze(PatientProfile(), [5.6, 5.8, 6.0])

def test_out_of_range_rejected():
    with pytest.raises(ValueError): orch().analyze(PatientProfile(), [100, 700])
    with pytest.raises(ValueError): orch().analyze(PatientProfile(), [])

def test_scenario_4_doses():
    r = run("4")   # policy giả luôn đề xuất 2.0
    assert (r.results["medication"].details["correction"], r.results["medication"].details["meal"]) == (0.5, 5.0)

def test_scenario_5_emergency_keeps_drug_review():
    r = run("5")
    assert r.plan.startswith("KHẨN CẤP") and "nutrition" not in r.results
    assert r.results["medication"].details["correction"] is None
    assert any("gliclazide" in a.message.lower() for a in r.alerts)

@pytest.mark.parametrize("key", SCENARIOS)
def test_auditor_silent_on_demo(key):
    assert run(key).audit_findings == []

def test_auditor_catches_forced_violation():
    results = {"medication": AgentResult("m", "", {"correction": 2.0, "meal": 3.0, "rules_fired": []})}
    assert {"A1", "A2"} <= {f.rule for f in audit(ctx(readings=flat(60)), results)}
    assert results["medication"].details["correction"] == 0

def test_gate_conflict_removed_by_a5():
    results = {"nutrition": AgentResult("n", "", {}, ["Đi bộ sau ăn"])}
    audit(ctx(T1(), readings=flat(265), code="T1"), results)
    assert not results["nutrition"].actions

def test_random_profiles_never_violate():
    # 250 hồ sơ ngẫu nhiên, default_rng(7), qua analyze: audit_findings rỗng, và
    # liều > 0 chỉ xuất hiện khi on_insulin và G >= 70
    ...

def test_every_report_has_disclaimer_and_sources():
    r = run("1"); assert "không thay thế bác sĩ" in r.disclaimer and r.sources
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt `orchestrator.py`, `auditor.py`, `scenarios.py`**

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat: orchestrator, safety auditor and demo scenarios`

---

### Task 9: LLM tuỳ chọn có guardrail số liệu

Đọc skill `claude-api` trước khi viết `llm.py`.

**Files:**
- Create: `diabetes_mas/llm.py`, `diabetes_mas/guardrail.py`
- Modify: `diabetes_mas/kb/rag.py` (thêm `RAG.answer`), `diabetes_mas/agents/orchestrator.py` (thêm `_synthesize`)
- Test: `tests/test_llm_guardrail.py`

**Interfaces:**
- Produces:
  - `unsupported_numbers(llm_text, source_text) -> list[str]`: số đi kèm `đv|đơn vị|mg/dL|mg|g|kcal|phút|%`; chuẩn hoá `,` và `.`
  - `is_grounded(llm_text, source_text) -> (bool, list[str])`
  - `LLM(client=None, model=env LLM_MODEL hoặc "claude-opus-5-5")`, `.enabled` (có `ANTHROPIC_API_KEY` hoặc có `client`), `.complete(system, user) -> str|None`; lỗi thì trả `None` và lưu `.last_error`, không bao giờ trả chuỗi lỗi như nội dung
  - `RAG.answer(question, llm=None) -> {"answer", "evidence", "mode": "llm"|"extractive"}`
  - `Orchestrator._synthesize(report) -> (text|None, note|None)`, `note` giải thích vì sao không có bản tổng hợp

- [ ] **Step 1: Viết test thất bại**

```python
def test_invented_dose_rejected():
    ok, bad = is_grounded("Hãy tiêm 9 đơn vị", "liều 2,0 đv")
    assert not ok and bad

def test_same_number_other_format_passes():
    assert is_grounded("tiêm 2.0 đv", "liều 2,0 đv")[0]

def test_llm_error_returns_none():
    llm = LLM(client=RaisingClient())
    assert llm.complete("s", "u") is None and llm.last_error

def test_answer_without_llm_is_extractive():
    assert RAG(load_kb()).answer("tụt đường")["mode"] == "extractive"

def test_answer_drops_ungrounded_llm():
    a = RAG(load_kb()).answer("tụt đường", llm=FakeLLM("tiêm 99 đơn vị"))
    assert a["mode"] == "extractive" and "99" not in a["answer"]

def test_synthesis_note_when_disabled():
    r = run("1")
    assert r.synthesis is None and "LLM" in r.synthesis_note
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_llm_guardrail.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt**

- [ ] **Step 4: Chạy test**

Run: `.venv/bin/pytest tests/test_llm_guardrail.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat: optional LLM with numeric guardrail`

---

### Task 10: API và giao diện web

**Files:**
- Create: `diabetes_mas/api.py`, `diabetes_mas/web/index.html`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Orchestrator`, `RAG.answer`, `SCENARIOS`, `Report.to_dict`
- Produces:
  - `create_app(orchestrator=None, rag=None) -> FastAPI`; `app = create_app()` dựng từ artifact thật khi khởi động qua `uvicorn diabetes_mas.api:app --host 127.0.0.1`
  - `POST /analyze` nhận `{profile, readings, unit: "mg/dL"|"mmol/L", planned_carbs, minutes_since_insulin, last_insulin_units, reading_age_min, meal_type}`, trả `Report.to_dict()`. `mmol/L` nhân 18,016 trước khi phân tích. `ValueError` → 422 kèm thông điệp.
  - `POST /ask` nhận `{question}`, trả `RAG.answer`
  - `GET /scenarios`, `GET /health` (`{"kb", "dl", "llm"}`), `GET /` (trang web)
  - Giới hạn Pydantic, `allow_inf_nan=False`: age 1–120, weight 20–300, height 100–250, hba1c 3–20, egfr 0–200, isf 5–200, icr 1–100, carbs 0–300, readings 1–288 phần tử

- [ ] **Step 1: Viết test thất bại**

```python
def client(): return TestClient(create_app(fake_orch(), RAG(load_kb())))

def test_analyze_ok():
    r = client().post("/analyze", json=body(readings=[100]))
    assert r.status_code == 200 and "disclaimer" in r.json()

def test_mmol_unit_converted():
    r = client().post("/analyze", json=body(readings=[5.6]*12, unit="mmol/L"))
    assert r.status_code == 200 and r.json()["results"]["glucose"]["details"]["current"] == pytest.approx(100.9, abs=.1)

@pytest.mark.parametrize("bad", [
    {"readings": []}, {"readings": [5.6, 5.8]},
    {"profile": {"weight_kg": 0}}, {"profile": {"isf": 0}}, {"profile": {"age": -1}},
])
def test_bad_input_is_422(bad):  # Review Focus 5
    assert client().post("/analyze", json=body(**bad)).status_code == 422

def test_nan_reading_is_422():
    raw = '{"profile": {}, "readings": [NaN]}'
    r = client().post("/analyze", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422

def test_ask_empty():
    assert "không tìm thấy" in client().post("/ask", json={"question": " "}).json()["answer"].lower()

def test_scenarios_and_index():
    assert len(client().get("/scenarios").json()) == 8
    assert "Phân tích" in client().get("/").text
```

- [ ] **Step 2: Chạy test**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Cài đặt `api.py`**

- [ ] **Step 4: Cài đặt `index.html`**

Một file HTML, JS thuần. Bố cục: trái là form hồ sơ + chọn ca mẫu + đơn vị; phải là số đường huyết hiện tại, thanh khoảng mục tiêu, thẻ phân loại/nguy cơ, cảnh báo theo mức, biểu đồ SVG (đo + dự báo), thẻ từng agent, thực đơn, liều tham khảo, mục "Auditor đã can thiệp" (chỉ hiện khi có), nguồn, câu miễn trừ. Góc phải dưới là khung hỏi đáp. Mọi dữ liệu đưa vào DOM dùng `textContent`, không dùng `innerHTML` với dữ liệu người dùng; input có `label`, màu cảnh báo kèm chữ.

- [ ] **Step 5: Chạy test**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 6: Chạy thật trên trình duyệt**

Run: `.venv/bin/uvicorn diabetes_mas.api:app --host 127.0.0.1`, mở `http://127.0.0.1:8000`, chạy lần lượt 8 ca mẫu và một câu hỏi "tụt đường phải làm gì".
Expected: kết quả khớp bảng kịch bản (ca 4 lúc này dùng RL thật, liều có thể khác 0,5), console không lỗi. Chụp màn hình để đưa vào báo cáo.

- [ ] **Step 7: Commit** — `feat: FastAPI and single-page web UI`

---

### Task 11: Số liệu, README và tổng duyệt

**Files:**
- Create: `diabetes_mas/eval/report_metrics.py`, `README.md`, `docs/metrics.md` (sinh tự động)
- Test: toàn bộ `tests/`

**Interfaces:**
- Consumes: `artifacts/meta.json`, `rl_insulin.evaluate`, `rag_eval`, kết quả pytest
- Produces: `python -m diabetes_mas.eval.report_metrics`, ghi `docs/metrics.md` gồm bảng phân loại + baseline, dự báo + baseline, RL, RAG, và ghi chú giới hạn của từng bảng

- [ ] **Step 1: Cài đặt `report_metrics.py`**

Thiếu artifact thì in dòng "chưa đo" chứ không dừng.

- [ ] **Step 2: Sinh số liệu**

Run: `.venv/bin/python -m diabetes_mas.eval.report_metrics`
Expected: `docs/metrics.md` có đủ 4 bảng, không còn "chưa đo".

- [ ] **Step 3: Viết `README.md`**

Gồm: sơ đồ bảng → module (bảng ở mục Đặc tả), lệnh cài, tải dataset, lệnh huấn luyện, chạy web, chạy test, giới hạn.

- [ ] **Step 4: Chạy lại từ đầu trên bản sạch**

Run: `git clone . /tmp/demo-check && cd /tmp/demo-check && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pytest -q`
Expected: toàn bộ PASS (test không cần `data/` hay artifact).

- [ ] **Step 5: Commit** — `docs: metrics report and README`

---

## Kịch bản trình bày trước thầy (5–7 phút)

1. Mở bảng đối chiếu sơ đồ thầy → module (30 giây).
2. Ca 1 → ca 3: Phân loại và lưới an toàn ADA (khối Deep learning → Phân loại).
3. Ca 4: chỉ biểu đồ dự báo, liều RL đề xuất rồi rule trừ IOB (khối Dự đoán, RL + Rule).
4. Ca 5: chế độ khẩn cấp, orchestrator đổi kế hoạch (khối Agent).
5. Ca 7: thuốc chống chỉ định và thuốc thay thế (khối KB).
6. Khung hỏi đáp "tụt đường phải làm gì", mở nguồn trích dẫn (khối RAG).
7. Chốt bằng `docs/metrics.md`: baseline so với mô hình và giới hạn của hệ thống.
