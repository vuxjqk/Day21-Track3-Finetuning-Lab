# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Phán quyết bị đảo khi tôi chạy đủ tập eval. Lần chạy thử với `EVAL_LIMIT=8` cho **PASSED**
(regression Δ = 0.000). Lần chạy đủ (50 target, 15 regression) cho **FAILED** (regression Δ = −0.113).
Cùng một adapter, cùng một mã, chỉ khác số câu được chấm. Điều ngạc nhiên thứ hai: cả 6 lỗi
của bản fine-tune đều là cùng một trường (`urgency`) trên cùng một cụm "Khi nào tiện". Trong
tập train, cụm này xuất hiện 35 lần với nhãn `thap` nhất quán, vậy mà model vẫn không học được.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Về máy, NB4 là lâu nhất: 1347 s (22.4 phút) cho ba run đối chứng, rồi tới NB3 (479 s). Điều này khớp dự đoán
vì README đã nói NB4 nặng nhất, dù thực tế nhanh hơn ước lượng 45–60 phút. Phần tôi *không*
dự đoán là phải chạy lại NB2 + NB5: lần đầu tôi để `EVAL_LIMIT=8` cho nhanh, và `make verify` chặn
lại vì đó là chế độ thử, không nộp được. Tôi cũng mất thời gian ở một lỗi giả: `verify` trên Windows
báo tập eval "bị sửa", trong khi thật ra Git chỉ đổi xuống dòng LF thành CRLF.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin rằng train loss thấp hơn nghĩa là adapter tốt hơn, và tăng rank là cách tăng chất
lượng. Bảng NB4 cho thấy `attn_only` (r=283) có loss thấp nhất (0.539 so với 0.629) nhưng chỉ hoà
`correct` trên target (0.97 = 0.97). `wrong_lr` có loss giảm đều suốt 30 step mà target bằng 0.
Tôi cũng từng nghĩ "fine-tune thắng prompt thì dùng fine-tune". Giờ tôi biết thắng trên tác vụ
(+0.205) vẫn có thể là một bản không được deploy, vì nó làm hỏng năng lực chung.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code (Anthropic) để: đọc README/rubric và hướng dẫn tôi các bước chạy trên Colab;
giải nén `results/`; **soạn toàn bộ `submission/REPORT.md` và file reflection này** từ số liệu trong
`results/`; đối chiếu dự đoán với nhãn đúng để tìm ra mẫu lỗi "Khi nào tiện"; chẩn đoán lỗi
checksum CRLF; và đóng gói file nộp. Việc chạy pipeline trên Colab là do tôi tự làm.

Chỗ AI sai hoặc có thể sai: sau lần chạy thử 8 mẫu, nó tóm tắt kết quả là "PASSED, regression Δ +0" và
gợi ý xu hướng có thể giữ nguyên. Lần chạy đủ cho kết quả ngược lại, nên kết luận từ dữ liệu
nhỏ không đáng tin. Ngoài ra, NB2 không lưu dự đoán từng mẫu của baseline (b). AI không có dữ liệu
đó nên để trống cột (b) trong bảng định tính thay vì đoán. Tôi đã đọc lại report và đối chiếu các
con số với `results/`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng bộ eval trước khi train: tập target của khách, **cộng một tập regression đủ lớn**
(không phải 15 câu) cho những năng lực họ không được phép mất, và đo baseline prompt tốt nhất
trên đó. Lab này cho thấy prompt tối ưu đã đạt 0.765 mà không cần train, và một tập regression
nhỏ có thể cho kết luận ngược hẳn. Nếu prompt đã đủ tốt cho khách, có thể không cần fine-tune.
Nếu vẫn fine-tune, tôi sẽ trộn 1–5% dữ liệu replay ngay từ đầu.
