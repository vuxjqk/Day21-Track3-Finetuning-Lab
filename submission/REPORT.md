# Lab 21 — Evaluation Report

**Họ tên**: Trần Anh Vũ  **MSSV**: 2A202602570  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 (sm_75, 14.6 GB) — Colab Free, fp16`

> Mọi con số dưới đây lấy trực tiếp từ các file trong `results/` (lần chạy đầy đủ,
> `EVAL_LIMIT` không đặt: 50 mẫu target, 15 mẫu regression).

---

## 1. Setup

**Lựa chọn và lý do.** Tôi giữ base model và dataset mặc định của tier T4:

- **Model `unsloth/Qwen3.5-4B`**: là model lớn nhất vừa train LoRA 16-bit trên T4 16 GB
  (đo được peak 8.78 GB). Đây là model có chế độ thinking (`<think>`), nên kiểm tra
  template ở NB1 có ý nghĩa thật.
- **Dataset 250 ticket CSKH → JSON triage 4 trường**: bài này chấm được hoàn toàn khách
  quan (so khớp từng trường), không cần LLM judge. Nhờ vậy mọi chênh lệch giữa các run là
  chênh lệch thật chứ không phải nhiễu của người chấm.

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON `{intent, urgency, product, sentiment}` (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (mặc định tier T4). p95 đo được là **98** token, max 101, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch = **30 optimizer step** (batch 1 × grad-accum 16 = batch hiệu dụng 16 < 32) |
| Precision | fp16 + GradScaler (T4 không có bf16) |

**Vì sao giữ `max_length=1024` dù p95 chỉ 98?** Mẫu dài nhất chỉ có 101 token, nên cả
1024 lẫn 256 đều **không cắt cụt mẫu nào**. Với `packing=False` và batch 1, mỗi mẫu được
pad tối đa tới độ dài của chính nó chứ không tới `max_length`. Vì vậy đổi sang 256 không
tiết kiệm được VRAM hay thời gian, và kết quả train giống hệt. Tôi giữ giá trị của tier để
mọi run (NB3 lẫn NB4) dùng đúng một cấu hình. Nếu bật `packing=True` thì `max_length` mới
quyết định độ dài chuỗi ghép, và lúc đó tôi sẽ đặt 256 theo p95.

**Template có giữ khối `<think>` không?** **Có** — `verdict: "reasoning preserved — safe
to train on traces"` *(results/template_check.json)*. Chuỗi render của ví dụ thử
`2+2?` vẫn còn nguyên `<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>`. Vì vậy
không cần xử lý gì thêm. Lưu ý: dữ liệu train của tôi có khối `<think>` **rỗng** (xem mục 2),
nên model học trả lời không suy luận — đó là lý do `valid_trace_rate = 0.0` ở mục 5.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (giải mã ngược từ `labels != -100`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (không tính loss): toàn bộ `<|im_start|>system ... <|im_start|>user Alo shop,
mình đặt balo laptop ... <|im_end|> <|im_start|>assistant <think>`. Như vậy câu hỏi và
system prompt bị che, chỉ JSON trả lời và token kết thúc `<|im_end|>` được giám sát. Token
`<|im_end|>` nằm trong loss là điều cần thiết để model học cách dừng — nhờ vậy `format = 1.0`.
So sánh: chế độ `everything` giám sát 94/94 token (100%) và sẽ dạy model viết lại cả câu hỏi.
Trên toàn tập train, tỉ lệ giám sát là 9014 / 20951 token (43.0%).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3292.1 |
| (b) base + optimized prompt | **0.765** | 0.791 | 1.000 | 1019.7 |
| (c) LoRA fine-tune | **0.970** | **0.678** | 1.000 | 1408.0 |

*(results/baselines_frozen.json, results/verdict.json — n = 50 target, 15 regression)*

**(b) có thật sự mạnh hơn (a) không?** **Có**, rất rõ: target 0.000 → 0.765, format
0.000 → 1.000. Với prompt ngắn, base model không trả về JSON đúng 4 khoá lần nào
(format = 0), lại sinh dài nên latency cao gấp ~3 lần (3292 ms so với 1020 ms). Prompt tối ưu liệt
kê rõ tập giá trị cho từng trường và yêu cầu chỉ trả JSON, nên model base đã làm đúng
~3/4 số trường mà chưa cần train.

**Có sửa `OPTIMIZED_PROMPT` không?** **Không.** SHA prompt vẫn là `719e74d3b6232053`
(`make verify`: "baseline (b) prompt unmodified"). Tôi giữ nguyên để mốc (b) là mốc đóng
băng trước khi train, không bị chỉnh sau khi đã thấy kết quả fine-tune.

Ghi chú: regression của (a) và (b) bằng nhau (0.791) vì bộ câu hỏi phổ thông được hỏi
không kèm system prompt — đó đúng là năng lực gốc của base model.

---

## 4. Giải phẫu cấu hình sai (NB4)

Cả bốn run dùng **cùng 30 step**, cùng dữ liệu, cùng seed 42. Mỗi run chỉ đổi **một biến** so với `correct`:

| Run | biến bị đổi | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | — (cấu hình chuẩn) | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6288 | **0.97** | 419.6 | 8.78 |
| `attn_only` | vị trí adapter | q,v (2 module) | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5392 | **0.97** | 279.8 | 8.79 |
| `wrong_lr` | learning rate | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.00** | 415.6 | 8.78 |
| `qlora` | lượng tử hoá base 4-bit | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 487.5 | 3.86 |

*(results/runs.csv, results/autopsy.json)*. `attn_only` lệch ngân sách tham số chỉ 0.025%
(32,456,704 so với 32,464,896), nên đây là phép so sánh *vị trí*, không phải so sánh *ngân sách*.

Thứ tự theo **train loss**: `attn_only` (0.539) < `correct` (0.629) < `qlora` (0.706) < `wrong_lr` (1.570).
Thứ tự theo **target**: `correct` = `attn_only` (0.97) > `qlora` (0.94) > `wrong_lr` (0.00).
→ Hai thứ tự **khác nhau ở vị trí dẫn đầu**: loss nói `attn_only` tốt nhất, còn điểm tác vụ nói nó chỉ hoà.

**4.1 — `attn_only` so với `correct`.**
Trên tập target, `attn_only` **hoà** `correct` (0.97 = 0.97, format đều 1.0), dù chỉ gắn
adapter vào 2 loại module (q, v) thay vì 12. Thứ tự này **không giống** thứ tự theo train
loss. Theo loss thì `attn_only` thắng rõ (0.539 so với 0.629), và loss của nó cũng giảm nhanh hơn
ngay từ step 10 (0.833 so với 1.385). Nếu chỉ nhìn cột loss, tôi sẽ kết luận sai rằng "đặt ở attention với
rank cao là tốt hơn". Loss thấp hơn ở đây phản ánh việc ghi nhớ 225 mẫu train nhanh hơn,
không phải làm tác vụ tốt hơn. Về *rank* so với *vị trí*: khi đã khớp ngân sách tham số, việc dồn hết
vào rank (r=283 trên q,v) không đem lại thêm điểm nào. Ngược lại, trên tác vụ hẹp và dễ này
(base + prompt đã đạt 0.765), vị trí cũng không phải đòn bẩy, vì cả hai cách đều chạm trần ~0.97.
Thí nghiệm này **không chứng minh được** all-linear tốt hơn attention-only trên bài này. Nó
chỉ chứng minh rằng tăng rank không thay được cho vị trí, và rằng tác vụ chưa đủ khó để
phân biệt hai cách đặt adapter. Một lợi ích phụ đo được: `attn_only` train nhanh hơn
(279.8 s so với 419.6 s) và sinh nhanh hơn (908 ms so với 1408 ms), vì chỉ có 2 module LoRA chạy kèm
khi chưa merge.

**4.2 — `wrong_lr` (LR 1e-5 thay vì 1e-4).**
Hai run cùng xuất phát ở loss 2.163. `correct` giảm xuống 0.145 sau 1 epoch và 0.027 ở cuối.
`wrong_lr` chỉ giảm chậm: 2.066 → 1.606 → 1.326 → 1.141 → 1.119. Mean token accuracy của nó
dừng ở ~0.79 so với 0.995. Đường loss của `wrong_lr` **vẫn đang đi xuống đều**, trông hoàn
toàn "khoẻ mạnh". Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận rằng "model đang học,
chỉ cần train thêm epoch", hoặc tệ hơn là "dữ liệu khó, cần rank lớn hơn". Thực tế, trên
tác vụ nó đạt **target 0.00 và format 0.00**: không trả về nổi một JSON hợp lệ, và sinh dài
tới mức latency lên 5317 ms. Một con số LR sai một bậc (thang full fine-tune áp lên LoRA) đủ
để xoá sạch kết quả, mạnh hơn hẳn mọi khác biệt về vị trí hay rank ở 4.1.

**4.3 — `qlora` (base 4-bit).**
QLoRA giảm peak VRAM từ 8.78 GB xuống **3.86 GB (−56%)**. Cái giá là: target giảm từ 0.97
xuống 0.94 (−0.03, tức thêm khoảng 6 trường sai trên 200 trường), train chậm hơn 16% (487.5 s
so với 419.6 s), và sinh chậm hơn 28% (1800 ms so với 1408 ms) vì phải giải lượng tử trọng số ở mỗi lượt.
Trên T4 16 GB, bản 16-bit đã vừa (8.78 GB). Như vậy QLoRA tốn cả độ chính xác lẫn tốc độ
mà không mở khoá được gì, nên số đo của tôi **ủng hộ** khuyến nghị "không dùng QLoRA cho
dòng model này" *khi phần cứng đủ chỗ cho 16-bit*. Tuy vậy, mức tụt 0.03 là nhỏ. Nếu
GPU chỉ có 6–8 GB thì QLoRA vẫn là lựa chọn hợp lý, vì đổi 3 điểm target lấy việc chạy được.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = −0.113` · `valid_trace_rate = 0.00`
Lý do của gate: *"general capability regressed by 0.113 (tolerance 0.020)"*.

**Diễn giải.** Bản fine-tune làm đúng việc được dạy: target tăng từ 0.765 lên 0.970 (+0.205),
format giữ 1.0, và nó làm được điều này **chỉ với prompt ngắn** — hành vi đã chuyển vào trọng
số. Nhưng nó trượt cổng vì năng lực phổ thông giảm từ 0.791 xuống 0.678, vượt xa ngưỡng cho
phép 0.02. Đây là dấu hiệu quên thảm hoạ (catastrophic forgetting) điển hình: 30 step với
LR 1e-4 trên 225 mẫu **cùng một dạng duy nhất** (ticket → JSON) đã kéo phân phối đầu ra về
phía tác vụ hẹp. Loss train xuống tới 0.027 và entropy 0.017 cho thấy model đã gần như học thuộc
một khuôn trả lời. Tập train không có mẫu phổ thông nào để giữ lại hành vi trả lời câu hỏi
thường, nên không có gì kéo model ngược lại.

Cần đọc con số này một cách thận trọng. Tập regression chỉ có 15 câu, chấm bằng keyword
recall, nên −0.113 tương đương mất khoảng 1–2 câu trả lời đúng. Sai số thống kê vì vậy lớn.
Tuy nhiên, chiều của kết quả khớp với cơ chế đã biết, và cổng được đặt trước khi chạy. Nới
ngưỡng sau khi thấy kết quả là đúng loại gian lận mà lab cảnh báo. Tôi giữ nguyên phán quyết FAILED.

Kết luận cho bài toán: fine-tune **có** đòn bẩy thật trên tác vụ (+0.205 so với một prompt đã
tốt), nên câu trả lời không phải "không cần fine-tune". Câu trả lời là "fine-tune theo cách
này chưa an toàn để thay base model dùng chung". Bước sửa theo thứ tự chẩn đoán của NB5
là trộn 1–5% dữ liệu phổ thông (replay) vào tập train, rồi đo lại cả hai cột target và regression.

`valid_trace_rate = 0.0` là kết quả **dự kiến**, không phải lỗi: dữ liệu train có khối
`<think>` rỗng (mục 2), nên model học bỏ qua suy luận. Với tác vụ phân loại ngắn như này, điều
đó chấp nhận được, nhưng nếu muốn giữ chế độ thinking thì phải train trên trace thật (bonus B3).

---

## 6. Định tính — bắt buộc có cả ca THUA

Nhãn đúng lấy từ `data/eval_target.jsonl`, dự đoán (c) và điểm lấy từ `results/qualitative.json`.

> **Hạn chế:** NB5 chỉ lưu dự đoán từng mẫu của bản fine-tune. NB2 chỉ lưu điểm tổng hợp
> của (b), nên tôi **không có** dự đoán từng mẫu của (b) để đặt cạnh. Vì vậy "thắng/thua" dưới
> đây là so với **nhãn đúng**. Để so trực tiếp với (b) theo từng mẫu, cần sửa NB2 để lưu `preds`.

| # | Ticket (rút gọn) | Nhãn đúng (intent / urgency / product / sentiment) | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | #27 "…bàn phím cơ… Cho tôi trả lại. **Ngay lập tức**. Mình vẫn tin tưởng shop." | doi_tra / cao / bàn phím cơ / tich_cuc | *(không lưu)* | doi_tra / cao / bàn phím cơ / tich_cuc — **1.00** | ✅ FT đúng cả 4 trường, kể cả sentiment tích cực dù ticket là yêu cầu trả hàng |
| 2 | #33 "…chuột không dây… Khi nào có tiền về. **Gấp**. Quá tệ." | hoan_tien / cao / chuột không dây / tieu_cuc | *(không lưu)* | hoan_tien / cao / chuột không dây / tieu_cuc — **1.00** | ✅ FT hiểu "khi nào có tiền về" là hoàn tiền dù không có chữ "hoàn" |
| 3 | #3 "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện**. Cảm ơn shop nhiều." | hoan_tien / **thap** / bình giữ nhiệt / tich_cuc | *(không lưu)* | hoan_tien / **trung_binh** / bình giữ nhiệt / … — **0.75** | ❌ **FT thua**: sai urgency |
| 4 | #39 "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện**. Quá tệ." | hoan_tien / **thap** / nồi chiên không dầu / tieu_cuc | *(không lưu)* | hoan_tien / **trung_binh** / nồi chiên không dầu / … — **0.75** | ❌ **FT thua**: sai urgency |
| 5 | #46 "…đèn bàn LED… Sai màu. **Khi nào tiện**. Shop hỗ trợ tốt." | san_pham_loi / **thap** / đèn bàn LED / tich_cuc | *(không lưu)* | san_pham_loi / **trung_binh** / đèn bàn LED / … — **0.75** | ❌ **FT thua**: sai urgency |

**Có mẫu chung nào ở các ca FT thua không?** **Có, và rất rõ.** Bản fine-tune chỉ sai đúng
6/50 ticket (#3, 5, 12, 39, 41, 46). **Cả 6 đều sai cùng một trường (urgency) theo cùng một
cách** (dự đoán `trung_binh` thay vì `thap`), và **cả 6 đều chứa cụm "Khi nào tiện"**. Đây cũng
chính là toàn bộ 6 ticket có cụm này trong tập eval. 6 × 1/4 trường = 1.5 điểm trên 50 mẫu, đúng
bằng phần 0.03 còn thiếu để đạt 1.0. Điều đáng chú ý: trong tập train, cụm "Khi nào tiện" xuất hiện
35 lần và **cả 35 lần đều gán `thap`**. Model vẫn không học được quy tắc này sau 30 step. Giả
thuyết của tôi là "Khi nào…" có dạng một câu hỏi về thời gian, nên tiên nghiệm của base
model kéo nó về mức trung bình, mạnh hơn tín hiệu từ dữ liệu. Đây là lỗi hệ thống chứ không phải
nhiễu ngẫu nhiên, nên có thể sửa có mục tiêu (thêm mẫu tương phản, hoặc nêu quy tắc này trong prompt).

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không deploy** bản fine-tune này để thay thế base model dùng chung. Về tác
vụ, nó thắng rõ: target 0.970 so với 0.765 của prompt tối ưu, format tuyệt đối, và các lỗi còn lại tập
trung vào đúng một quy tắc urgency. Nhưng cổng hồi quy FAILED, vì năng lực phổ thông giảm
0.113, gấp hơn 5 lần ngưỡng 0.02. Nguyên nhân là tập train chỉ có một dạng tác vụ và không
có replay, nên 30 step với LR 1e-4 đủ để đẩy model về phía khuôn JSON. Nó cũng chậm hơn (b)
khoảng 390 ms mỗi yêu cầu khi chưa merge adapter. Nếu hệ thống *chỉ* dùng model này cho triage
ticket và không bao giờ cho hỏi đáp chung, việc deploy có thể bàn lại. Kể cả khi đó, tôi vẫn sẽ
merge adapter (NB6) để lấy lại latency và sửa lỗi "Khi nào tiện" trước. Đường đi đúng là train lại
với 1–5% dữ liệu phổ thông trộn vào, rồi chạy lại đúng cổng này.

Về đòn bẩy thật sự trong lab: **learning rate và mask** quyết định có hay không có kết quả. LR sai
một bậc cho target 0.00. Mask đúng giúp format đạt 1.0 nhờ học `<|im_end|>`. **Vị trí adapter và
rank** không tạo khác biệt trên tác vụ này: `attn_only` hoà `correct` ở cùng ngân sách. **Chất lượng
và độ đa dạng dữ liệu** quyết định phần còn lại: lỗi duy nhất còn sót là một cụm từ, và việc thiếu
replay là nguyên nhân trượt cổng. **Chỉ số thay thế (train loss) sai ở cả hai đầu**: nó xếp
`attn_only` lên trên `correct` và làm `wrong_lr` trông như đang học tốt.

**Ba điều tôi học được:**
1. **Train loss xếp hạng sai.** Bảng NB4 xếp `attn_only` (0.539) tốt hơn `correct`
   (0.629), nhưng trên tập target hai run hoà nhau ở 0.97. `wrong_lr` có đường loss giảm đều suốt
   30 step nhưng ra target 0.00 và format 0.00. Từ giờ tôi không kết luận gì về một adapter
   khi chưa chấm nó trên tác vụ.
2. **Thắng trên tác vụ không có nghĩa là deploy được.** Lần chạy thử với 8 mẫu cho PASSED
   (regression Δ = 0.000), còn lần chạy đủ 15 câu regression cho FAILED (Δ = −0.113).
   Nếu nộp theo lần chạy thử, tôi đã báo cáo một kết luận sai. Kích thước tập đánh giá quyết định
   việc tôi có *thấy* được sự quên hay không.
3. **Lỗi của model thường có cấu trúc, và chỉ đọc từng mẫu mới thấy.** Con số 0.97 trông như
   "gần hoàn hảo, còn lại là nhiễu". Đối chiếu nhãn từng ca thì cả 6 lỗi là cùng một trường, cùng một
   cụm "Khi nào tiện", dù cụm này có 35 lần trong train với nhãn nhất quán. Điểm tổng hợp che mất
   điều này. Bảng định tính mới cho biết phải sửa gì.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Trộn ~5% câu hỏi phổ thông (replay) vào tập train, giữ nguyên 30 step, rồi đo lại cổng hồi quy
  để kiểm tra xem regression có về trong ngưỡng 0.02 mà target vẫn giữ trên 0.765 không.
- Sửa NB2 để lưu dự đoán từng mẫu của (b), nhằm so (b) với (c) trực tiếp theo từng ticket.
- Thêm mẫu tương phản cho "Khi nào tiện" và quét LR (5e-5, 1e-4, 2e-4) để xem lỗi urgency là do
  thiếu step hay do tiên nghiệm của base model.
- Chạy NB6 (merge) để đo xem latency của fine-tune có xuống dưới mức 1020 ms của (b) không, vì
  prompt của (c) ngắn hơn nhiều.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/vuxjqk/lab21-qwen35-triage-vi

**B5.** Adapter `correct` (`adapter_config.json` + `adapter_model.safetensors`, 129.9 MB) được
push công khai lên HuggingFace Hub, kèm model card ghi cấu hình train, bảng kết quả và phán quyết
FAILED của cổng hồi quy. Các bài thưởng B1–B4 không làm trong lần nộp này.

**Hình thức nộp:** Option B (GitHub + HuggingFace Hub). Xem `LINKS.md` ở thư mục gốc repo.
