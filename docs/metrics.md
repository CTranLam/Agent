# Kết quả đo

Sinh tự động bởi `python -m diabetes_mas.eval.report_metrics`.

## Phân loại (DiabetesNet)

100,000 dòng, còn 96,146 sau khi loại trùng; ngưỡng 0.093 chọn trên validation cho recall ≥ 90%. Số đo trên tập test.

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.901 | 0.470 | 0.916 | 0.621 | 0.977 | 0.884 |

| Mô hình | ROC-AUC |
|---|---|
| Chỉ HbA1c | 0.848 |
| Chỉ đường huyết | 0.792 |
| Hồi quy logistic 13 đặc trưng | 0.964 |
| DiabetesNet | 0.977 |

**Giới hạn:** nhãn dataset gần như do hai ngưỡng HbA1c và đường huyết quyết định, nên AUC cao không phải khả năng dự báo trên người thật. Mạng sâu chỉ hơn logistic một chút.

## Dự báo đường huyết 30 phút

45 chuỗi test, 10,272 cửa sổ, 483 cửa sổ có hạ đường huyết sắp xảy ra.

| Phương pháp | MAE (mg/dL) | Recall hạ đường huyết | Precision |
|---|---|---|---|
| Giữ nguyên giá trị cuối | 6.70 | 0.000 | n/a |
| Ngoại suy tuyến tính | 6.68 | 0.841 | 0.635 |
| GlucoseLSTM | 4.42 | 0.617 | 0.958 |
| min(LSTM, tuyến tính), hệ thống dùng | 4.81 | 0.843 | 0.636 |

**Giới hạn:** CGM MÔ PHỎNG do chính dự án tạo, các cửa sổ cùng chuỗi còn tương quan; chưa kiểm chứng trên người thật.

## Học tăng cường đề xuất liều

| Chính sách | TIR | Hạ đường huyết do liều | Trên 180 |
|---|---|---|---|
| Không tiêm | 36.1% | 0.00% | 58.4% |
| Công thức (G-110)/ISF | 85.0% | 0.71% | 8.8% |
| Công thức có xét xu hướng | 87.3% | 0.00% | 7.2% |
| RL (bảng Q) | 86.4% | 0.62% | 7.5% |

**Giới hạn:** simulator đồ chơi, 20.000 quyết định; chưa áp rule an toàn. RL không vượt công thức có xét xu hướng; giá trị của lớp này là kiến trúc RL đề xuất rồi rule phủ quyết.

## RAG

| Bộ truy xuất | Recall@3 | MRR |
|---|---|---|
| Đếm từ trùng | 0.548 | 0.526 |
| BM25 + bigram + đồng nghĩa | 0.968 | 0.958 |

31 câu hỏi. **Giới hạn:** Bộ câu hỏi do chính người viết KB soạn nên số đo có xu hướng lạc quan; bộ lọc từ chuyên môn ngắn được thêm sau khi thấy câu trượt.
