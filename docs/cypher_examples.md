# Knowledge Graph: lược đồ và câu Cypher mẫu

Tài liệu ôn vấn đáp cho câu 20–24 và 46. Mọi câu Cypher dưới đây có trong `diabetes_mas/kb/kg_queries.py`
và chạy được trên đồ thị do `python -m diabetes_mas.kb.neo4j_kb` nạp. In lại để chép vào Neo4j Browser:

```bash
.venv/bin/python -m diabetes_mas.kb.kg_queries --cypher
.venv/bin/python -m diabetes_mas.kb.kg_queries --patient "Ca 5"     # chạy và in kết quả
```

Trên giao diện web: tab **Đồ thị tri thức**, khung **Truy vấn nhiều bước trên đồ thị** (góc trên ghi chạy trên `neo4j` hay `bộ nhớ`).

## 1. Lược đồ (property graph)

| Nhãn nút | Thuộc tính chính | Nguồn dữ liệu |
|---|---|---|
| `Patient` | name, title, age, sex, demo | 8 ca mẫu giả lập trong `scenarios.py` |
| `Disease` | code, name | `kb/clinical.json` (12 bệnh: DM, T1, T2, GDM, PRE, HYPO, HYPER, DKA, HHS, NEPHRO, RETINO, FOOT) |
| `Symptom` | name | `kb/clinical.json`, lấy từ đoạn tóm tắt Vinmec |
| `RiskFactor` | code, name | `kb/clinical.json` |
| `Measurement` | code, name, unit | HbA1c, đường huyết đói, đường huyết CGM, eGFR, BMI, C-peptide |
| `Drug` | name, aliases, not_for_types, url | `kb/drugs.json` |
| `Condition` | name | bệnh lý nền: `egfr_lt_30`, `heart_failure`, `age_ge_75`, ... |
| `Recommendation`, `Source`, `Topic` | title, text / name, url / name | `knowledge_docs/*.md` |
| `Food`, `FoodCategory` | name, carb_g, gi, ... | `kb/foods.json` |

| Quan hệ | Từ → Đến | Thuộc tính |
|---|---|---|
| `HAS_MEASUREMENT` | Patient → Measurement | value |
| `HAS_RISK_FACTOR` | Patient → RiskFactor | |
| `HAS_CONDITION` | Patient → Condition | |
| `TAKES` | Patient → Drug | |
| `DIAGNOSED_WITH` | Patient → Disease | source |
| `INDICATES` | Measurement → Disease | low, high, rule (ngưỡng) |
| `INCREASES_RISK_OF` | RiskFactor → Disease | |
| `HAS_SYMPTOM` | Disease → Symptom | severity |
| `SUBTYPE_OF` | Disease → Disease | (T1, T2, GDM thuộc DM) |
| `MAY_LEAD_TO` | Disease → Disease | note (biến chứng) |
| `HAS_RECOMMENDATION` | Disease → Recommendation | seq |
| `CONTRAINDICATED_IN`, `CAUTION_IN` | Drug → Condition | note |
| `SUBSTITUTE_WHEN` | Drug → Drug | condition, why |
| `INTERACTS_WITH` | Drug → Drug | severity, message |
| `FROM`, `ABOUT` | Recommendation → Source / Topic | |
| `IN_CATEGORY` | Food → FoodCategory | |

## 2. Bộ ba chủ ngữ – vị ngữ – tân ngữ (câu 20)

| Chủ ngữ | Vị ngữ | Tân ngữ |
|---|---|---|
| Ca 7 | TAKES | pioglitazone |
| pioglitazone | CONTRAINDICATED_IN | heart_failure |
| pioglitazone | SUBSTITUTE_WHEN {condition: heart_failure} | empagliflozin |
| Hạ đường huyết | HAS_SYMPTOM | Vã mồ hôi |
| HbA1c | INDICATES {low: 6.5} | Đái tháo đường |
| Đái tháo đường type 1 | SUBTYPE_OF | Đái tháo đường |
| Đái tháo đường | MAY_LEAD_TO | Bệnh thận do đái tháo đường |
| Thừa cân, béo phì | INCREASES_RISK_OF | Đái tháo đường type 2 |

## 3. Tạo Patient, Disease và quan hệ rồi truy vấn lại (câu 22)

```cypher
MERGE (p:Patient {name: 'Bệnh nhân demo'}) SET p.age = 52, p.sex = 'F'
MERGE (d:Disease {code: 'T2'}) ON CREATE SET d.name = 'Đái tháo đường type 2'
MERGE (p)-[r:DIAGNOSED_WITH]->(d) SET r.since = 2020, r.source = 'bác sĩ'
RETURN p, r, d;

MATCH (p:Patient {name: 'Bệnh nhân demo'})-[r:DIAGNOSED_WITH]->(d:Disease)
RETURN p.name AS patient, p.age AS age, d.name AS disease, r.since AS since;
```

(Nút `Bệnh nhân demo` sẽ bị xoá khi nạp lại KB, vì lệnh nạp xoá mọi nút có nhãn của KB.)

## 4. Truy vấn nhiều bước nhảy (câu 24, 46)

### 4.1. Thuốc đang dùng chống chỉ định với bệnh lý nền nào, thay bằng thuốc gì

Tham số: $patient = "Ca 5"

```cypher
// Thuốc -> chống chỉ định/thận trọng -> bệnh lý nền CỦA BỆNH NHÂN -> thuốc thay thế an toàn với bệnh nhân
MATCH (p:Patient {name: $patient})-[:TAKES]->(d:Drug)-[ci:CONTRAINDICATED_IN|CAUTION_IN]->(c:Condition)<-[:HAS_CONDITION]-(p)
OPTIONAL MATCH (d)-[s:SUBSTITUTE_WHEN {condition: c.name}]->(alt:Drug)
  WHERE NOT EXISTS { MATCH (alt)-[:CONTRAINDICATED_IN|CAUTION_IN]->(:Condition)<-[:HAS_CONDITION]-(p) }
    AND NOT EXISTS { MATCH (p)-[:TAKES]->(alt) }
RETURN d.name AS drug, type(ci) AS relation, c.name AS condition, ci.note AS note,
       alt.name AS alternative, s.why AS why
ORDER BY drug, relation, condition, alternative
```

Kết quả:

| Thuốc | Quan hệ | Bệnh lý nền | Thay bằng | Lý do |
|---|---|---|---|---|
| gliclazide | CAUTION_IN | age_ge_75 | linagliptin | ít gây hạ đường huyết |
| gliclazide | CONTRAINDICATED_IN | egfr_lt_30 | linagliptin | ít gây hạ đường huyết, không cần chỉnh liều theo thận |

### 4.2. Chỉ số đo gợi ý bệnh gì theo ngưỡng

Tham số: $patient = "Ca 5"

```cypher
// Bệnh nhân -> chỉ số đo -> ngưỡng -> bệnh gợi ý (-> bệnh cha)
MATCH (p:Patient {name: $patient})-[m:HAS_MEASUREMENT]->(ms:Measurement)-[r:INDICATES]->(d:Disease)
WHERE (r.low IS NULL OR m.value >= r.low) AND (r.high IS NULL OR m.value < r.high)
OPTIONAL MATCH (d)-[:SUBTYPE_OF]->(parent:Disease)
RETURN ms.name AS measurement, m.value AS value, ms.unit AS unit, r.rule AS rule, d.name AS disease, parent.name AS parent
ORDER BY measurement, disease
```

Kết quả:

| Chỉ số | Giá trị | Ngưỡng | Gợi ý |
|---|---|---|---|
| HbA1c | 7.0 | HbA1c ≥ 6,5% (ADA) | Đái tháo đường |
| eGFR | 25.0 | eGFR < 60, gợi ý suy giảm chức năng thận | Bệnh thận do đái tháo đường |
| Đường huyết hiện tại (CGM) | 46.8 | Đường huyết < 70 mg/dL | Hạ đường huyết |

### 4.3. Yếu tố nguy cơ làm tăng nguy cơ bệnh nào

Tham số: $patient = "Ca 5"

```cypher
// Bệnh nhân -> yếu tố nguy cơ -> bệnh có nguy cơ tăng
MATCH (p:Patient {name: $patient})-[:HAS_RISK_FACTOR]->(rf:RiskFactor)-[:INCREASES_RISK_OF]->(d:Disease)
WITH d, rf ORDER BY rf.name
RETURN d.name AS disease, collect(rf.name) AS risk_factors, count(rf) AS n
ORDER BY n DESC, disease
```

Kết quả:

| Bệnh | Yếu tố nguy cơ |
|---|---|
| Biến chứng bàn chân | Mắc đái tháo đường từ 10 năm |
| Bệnh thận do đái tháo đường | Mắc đái tháo đường từ 10 năm |
| Bệnh võng mạc do đái tháo đường | Mắc đái tháo đường từ 10 năm |
| Hạ đường huyết | Dùng insulin hoặc sulfonylurea |
| Tiền đái tháo đường | Tuổi từ 45 |
| Đái tháo đường type 2 | Tuổi từ 45 |

### 4.4. Biến chứng cần phòng và khuyến nghị liên quan

Tham số: $patient = "Ca 4"

```cypher
// Bệnh nhân -> chẩn đoán -> (bệnh cha) -> biến chứng -> khuyến nghị -> nguồn  (4-5 bước nhảy)
MATCH (p:Patient {name: $patient})-[:DIAGNOSED_WITH]->(dx:Disease)-[:SUBTYPE_OF*0..1]->(g:Disease)-[:MAY_LEAD_TO]->(c:Disease)
WITH DISTINCT c
MATCH (c)-[h:HAS_RECOMMENDATION]->(r:Recommendation)-[:FROM]->(s:Source)
WITH c, h, r, s ORDER BY h.seq
RETURN c.name AS complication, collect(r.title) AS recommendations, collect(DISTINCT s.url) AS sources
ORDER BY complication
```

Kết quả:

| Biến chứng | Khuyến nghị |
|---|---|
| Biến chứng bàn chân | Chăm sóc bàn chân; Vận động khi có vấn đề bàn chân |
| Bệnh thận do đái tháo đường | Tầm soát biến chứng thận; Kiểm soát huyết áp và đường huyết khi có bệnh thận; Metformin và chức năng thận |
| Bệnh võng mạc do đái tháo đường | Tầm soát biến chứng mắt |
| Hạ đường huyết | Ngưỡng hạ đường huyết; Xử trí hạ đường huyết theo quy tắc 15-15; Xử trí khi hạ đường huyết nặng |
| Nhiễm toan ceton | Nhiễm toan ceton khi đường huyết cao; Dấu hiệu cấp cứu của tăng đường huyết |
| Tăng đường huyết | Xử trí tăng đường huyết; Dấu hiệu cấp cứu của tăng đường huyết |

### 4.5. Có triệu chứng thì nghĩ tới bệnh gì, nên làm gì

Tham số: $symptom = "Vã mồ hôi"

```cypher
// Triệu chứng -> bệnh -> khuyến nghị
MATCH (sy:Symptom {name: $symptom})<-[h:HAS_SYMPTOM]-(d:Disease)-[k:HAS_RECOMMENDATION]->(r:Recommendation)
WITH d, h, k, r ORDER BY k.seq
RETURN d.name AS disease, h.severity AS severity, collect(r.title) AS recommendations
ORDER BY disease
```

Kết quả:

| Bệnh | Mức | Khuyến nghị |
|---|---|---|
| Hạ đường huyết | nhẹ | Ngưỡng hạ đường huyết; Xử trí hạ đường huyết theo quy tắc 15-15; Xử trí khi hạ đường huyết nặng |

**Cách giải thích khi vấn đáp:** truy vấn 4.1 đi 4 bước nhảy `Patient → Drug → Condition ← Patient` rồi
`Drug → Drug` (thay thế), và tự loại thuốc thay thế nếu chính nó vướng bệnh lý nền của bệnh nhân (`NOT EXISTS {...}`).
Truy vấn 4.4 dùng đường đi độ dài thay đổi `[:SUBTYPE_OF*0..1]` để biến chứng chung của "Đái tháo đường" áp dụng
cho cả type 1 và type 2. Mỗi truy vấn có bản Python tương đương (`memory_query`); test
`tests/test_neo4j_kb.py::test_multi_hop_cypher_matches_memory` kiểm tra hai bản cho cùng kết quả trên 8 bệnh nhân.

**Giới hạn:** bệnh nhân là ca mẫu giả lập; ngưỡng và quan hệ do đồ án tóm tắt, chưa được bác sĩ thẩm định.
`MedicationAgent` khi phân tích vẫn dùng `interactions()` / `alternatives()` (1 bước nhảy); các truy vấn nhiều bước ở đây
phục vụ tra cứu và minh hoạ trên giao diện, chưa đưa vào luồng ra quyết định của agent.
