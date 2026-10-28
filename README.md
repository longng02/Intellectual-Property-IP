# AIP Update Pipeline

Pipeline hỗ trợ rà soát prior art cho claim sáng chế. Output là evidence có truy vết, **không** là kết luận pháp lý về tính mới, trình độ sáng tạo hay khả năng bảo hộ.

## Luồng chạy hiện tại

1. `acquire`: ghi nhận nguồn, PDF/metadata/prior art và SHA-256.
2. `ocr`: nhận artefact từ Tesseract `vie_best` (chính), PP-OCRv6 (hỗ trợ), hybrid selector, safe correction v4 và consensus marker repair v6/v7.
3. `corpus`: tạo corpus claim/tài liệu, metadata, hash và split theo family.
4. `features`: dịch VI sang EN bằng Qwen3-8B Q4 GGUF khi chưa có `text_en`; sau đó đọc feature artefact có truy vết do extractor đã chọn tạo ra.
5. `retrieve`: baseline BM25 trên toàn văn, lọc ngày công bố và family của hồ sơ.
6. `passages`: lấy các passage trong tài liệu đã truy xuất.
7. `evidence`: Qwen3.5-4B qua prompt evidence riêng chấm mỗi passage `0..3`, quy đổi `SUPPORTED` / `UNCLEAR` / `MISSING`.
8. `review`: Gemini 3.5 Flash Lite đọc bảng feature-evidence cho từng tài liệu Top 5, ghi nhận xét có trích passage; không đưa ra kết luận pháp lý.

## Trạng thái đánh giá: không gộp các benchmark

Các số dưới đây là **thí nghiệm tách rời**, chưa phải một phép đo chạy nối từ OCR đến evidence:

- Tách feature trên 100 claim: Gemini 3.5 Flash Lite đạt F1 feature chính xác **71,2%**, F1 ranh giới chính xác **80,9%**, ranh giới dung sai 5 ký tự **89,4%**. Qwen3-8B local lần lượt **35,3%**, **13,0%**, **70,9%**. Nhãn examiner chỉ là mốc tham khảo.
- Tìm tài liệu: trên corpus FiNE đóng gồm 3.100 publication được trích dẫn và 100 claim, BM25 toàn văn + BGE-M3 + RRF đạt Hit@200 **93%**, Hit@10 **62%**, MRR **0,391**. Đây không phải kết quả của BM25 scaffold trong thư mục này, cũng không phải đánh giá trên toàn bộ corpus 41k.
- Tìm passage: 52,2% Hit@10 trên 406 feature là phép **replay lọc section/description-first** sau xếp hạng, không phải kết quả full-text re-index mới; raw RRF tương ứng là 45,8%. Trên tập thăm dò 93 feature, E1/Qwen3.5-4B đạt 54/93 (**58,1%**) khi tính 5 fallback; trên 88 phản hồi hợp lệ là 51/88, so với baseline 47/88 và Gemini 50/88. Tập này nhỏ và tái sử dụng, nên chưa đủ để kết luận hơn hẳn độc lập.

Vì vậy, `AIP_update` mặc định dùng `features.mode=external_artifacts`: hãy nạp output lưu vết model/prompt/version đã được chọn. `rule` chỉ là fallback kỹ thuật cho test, không đại diện cho Gemini hoặc Qwen benchmark. Retrieval trong code cũng cố ý ghi `BM25 baseline only`; chưa được phép gắn các chỉ số BGE-M3+RRF ở trên vào một run mới.

## Chạy

```powershell
cd AIP_update
python -m aip_pipeline.cli init
python -m unittest discover -s tests -v
```

Để chạy đủ evidence, khởi động Ollama và cài `qwen3.5:4b`, rồi chạy:

```powershell
ollama serve
ollama pull qwen3.5:4b
python -m aip_pipeline.cli run --config config.json
```

Qwen3-8B-Q4_K_M chỉ được nạp để dịch khi claim chưa có `text_en`; cần có `llama-cpp-python` và model tại đường dẫn trong `config.json`. Không có model hoặc bản dịch, pipeline dừng thay vì thay thế âm thầm.

Để bật bước tổng hợp Gemini, đặt `review.enabled` thành `true` và cung cấp `GEMINI_API_KEY` qua biến môi trường. Kết quả ở `08_reviewer_report.jsonl`. Khi tắt, pipeline vẫn ghi các dòng `not_run` để phân biệt rõ chưa gọi Gemini với kết luận chuyên môn.

## Input tối thiểu

- `claims.jsonl`: `claim_id`, `document_id`, `text_vi`, `priority_date`, `family_id`.
- `prior_art.jsonl`: `publication_id`, `family_id`, `publication_date`, `title`, `abstract`, `text`.
- `ocr_artifacts.jsonl`: output thô Tesseract/PP-OCR theo `claim_id` (tuỳ chọn).
- `features.jsonl`: `claim_id`, `method`, `features`; mỗi feature nên giữ `feature_id`, `text`, `start_char`, `end_char`, model/prompt version nếu có.
