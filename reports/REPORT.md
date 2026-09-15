# Báo cáo Ngày 3 — Tracking Annotation

Họ tên / nhóm: `Nguyễn Văn Trọng`
Ngày: `15/9/2026`

---

## 1. Quá trình gán nhãn

| Mục | Giá trị |
| --- | --- |
| Công cụ | CVAT / khác: `CVAT Track Mode` (theo quy trình của lab; chưa có log phiên CVAT) |
| Thời gian gán `clip_02` (warm-up) | `chưa ghi nhận` phút |
| Thời gian gán `clip_01` | `chưa thực hiện / chưa ghi nhận` phút |
| Số track đã vẽ trong `clip_01` | `chưa có annotations/clip_01/gt.txt` |
| Số keyframe trung bình mỗi track | `chưa xác định; file MOT không lưu thông tin keyframe` |

Ba tình huống khó nhất khi gán clip này, và bạn xử lý thế nào:

1. `Với clip_02, xe bị che hoặc đi sát rìa ảnh: giữ cùng track_id khi còn nhận ra là cùng xe, chỉ vẽ phần nhìn thấy và để bbox chạm rìa ảnh khi cần.`
2. `Khi xe ra khỏi khung: dùng Outside tại frame xe biến mất, không để bbox treo ở các frame sau; nếu xe quay lại thì theo luật mặc định tạo track mới.`
3. `Khi hai xe chồng lấp/cắt nhau: kiểm tra chậm từng frame quanh transition, giữ ID theo quỹ đạo và appearance liên tục, không đổi ID chỉ vì hai bbox overlap.`

`Ba tình huống trên là cách xử lý theo guideline của lab; chưa có nhật ký frame/ID riêng của phiên annotation để xác nhận đây là ba ca khó thực tế của clip_01.`

## 2. Tự kiểm và kiểm chéo

Ba lượt tua bắt được gì (lượt 1 nhìn ID, lượt 2 frame đầu/cuối, lượt 3 frame giữa):

- Lượt 1: `Chưa có log tự kiểm. Cần kiểm từng xe có giữ một ID qua occlusion/crossing và không reuse ID cho xe khác.`
- Lượt 2: `Chưa có log tự kiểm. Cần kiểm frame bắt đầu, frame Outside và bbox treo sau khi xe rời khung.`
- Lượt 3: `Chưa có log tự kiểm. Cần kiểm drift của interpolation và độ khít bbox ở giữa hai keyframe.`

Kiểm chéo với: `chưa thực hiện`. Chi tiết ở `reports/review_partner.md`.
Số lỗi bạn tìm được trong bản của bạn ấy: `chưa có dữ liệu`. Số lỗi bạn ấy tìm được trong bản của bạn: `chưa có dữ liệu`.

Ca nào hai người quyết khác nhau, và luật nào còn thiếu trong `GUIDELINE_MINI.md`?

`Chưa có peer review nên chưa thể nêu ca bất đồng cụ thể. Luật cần ghi rõ hơn là cách phân biệt occlusion ngắn với việc xe thực sự ra khỏi khung: dưới ngưỡng 25 frame thì giữ ID nếu còn evidence liên tục; ra khỏi khung rồi quay lại mặc định tạo track mới.`

## 3. Pre-gold lock và chấm trước/sau rework

| Evidence | Giá trị |
| --- | --- |
| SHA-256 từ `evidence/pre-gold/clip_01/manifest.json` | `chưa có manifest.json` |
| Thời điểm khóa | `chưa khóa pre-gold` |
| Số row / frame / track trước khi mở reference | `chưa có pre-gold/clip_01/gt.txt` |

`Warm-up hiện có 246 row trong 60/60 frame, dùng 6 track ID 1-6. Đây là annotation warm-up, không phải pre-gold của clip_01.`

| | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bản pre-gold | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Sau rework | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): **chưa thể kết luận** vì chưa có `clip_01` và `eval_vs_gold.json`.

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi | Frame | ID | Đã sửa thế nào |
| --- | --- | --- | --- |
| Chưa có danh sách lỗi gold/review | N/A | N/A | Chưa thực hiện rework |
| Chưa có annotation clip_01 | N/A | N/A | Chưa có frame/ID để sửa |
| Chưa có pre-gold evidence | N/A | N/A | Chưa thể đối chiếu trước/sau |

## 4. Kết quả model: ByteTrack control vs ReID treatment

Cấu hình từ `outputs/model_run_config.json`:

| Mục | Giá trị |
| --- | --- |
| Python / ultralytics / torch / lap | `Chưa có model_run_config.json; notebook sẽ ghi runtime versions sau khi chạy. Notebook cài ultralytics==8.4.145 và lap==0.5.13.` |
| weights / hai tracker | `yolo26n.pt; ByteTrack: bytetrack.yaml; treatment: BoT-SORT + ReID dùng configs/trackers/botsort-reid.yaml` |
| conf / IoU / imgsz / classes | `0.25 / 0.70 / 960 / [2, 5, 7] (car, bus, truck)` |
| device | `Tự chọn "0" nếu CUDA khả dụng, nếu không là "cpu"; chưa có runtime output để xác nhận` |

| So sánh | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bạn vs gold | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ByteTrack control vs gold | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| BoT-SORT + ReID vs gold | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReID vs bạn | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

`Các file eval và hai file MOT model chưa được sinh ra, nên không được điền số metric giả.`

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt nặng lỗi ID?**

`Chưa thể xác định cao hơn hay thấp hơn vì chưa có eval_vs_gold.json. Về ý nghĩa, MOTA có thể cao nhưng IDF1 thấp khi detector vẫn phát hiện đúng số lượng xe, trong khi track_id bị đổi, bị phân mảnh hoặc gán nhầm giữa hai xe. MOTA dùng FN, FP và IDSW trong tử số: MOTA = 1 - (FN + FP + IDSW) / số đối tượng ground truth. Vì vậy mỗi lỗi identity chỉ bị tính như một thành phần lỗi tại frame và thường không làm MOTA giảm mạnh. IDF1 đo trực tiếp độ nhất quán của identity qua các frame nên nhạy hơn với đổi ID và fragment.`

**2. ByteTrack control và BoT-SORT + ReID treatment khác nhau thế nào ở IDF1, AssA và IDSW? Dẫn một frame sequence để giải thích treatment tốt hơn, tệ hơn hoặc không đổi đáng kể. Nhắc rõ đây không cô lập causal effect của ReID vì hai tracker implementation khác.**

`Chưa thể nêu chiều thay đổi hoặc frame sequence cụ thể vì chưa có eval_bytetrack_vs_gold.json, eval_reid_vs_gold.json và model outputs. Cách phân tích đúng là đọc IDF1/AssA/IDSW trước, sau đó chọn một đoạn frame có occlusion hoặc crossing: nếu ReID giữ đúng cùng ID thì IDF1 và AssA tăng, IDSW giảm; nếu appearance cue gây ghép nhầm thì ngược lại; nếu detector không mất xe và motion đã đủ rõ thì chênh lệch có thể không đáng kể. Đây chỉ là system comparison, không cô lập causal effect của ReID, vì ByteTrack và BoT-SORT là hai implementation tracker khác nhau, dù detector input được giữ cố định.`

**3. DetA, FP và FN đổi thế nào? Lỗi còn lại là detector hay association?**

`Chưa thể đọc chiều thay đổi khi chưa có eval JSON. DetA giảm cùng FP/FN tăng thường chỉ ra lỗi coverage/detection hoặc ngưỡng lọc: FP là bbox thừa, FN là xe bị bỏ sót. Nếu DetA, FP và FN gần như giữ nguyên nhưng AssA/IDF1 giảm và IDSW tăng, lỗi chính là association/identity. ReID có thể giúp association khi xe overlap hoặc vừa hiện lại, nhưng không thể tự sửa một detection bị thiếu; vì vậy thay đổi ở DetA/FP/FN không nên quy toàn bộ cho ReID.`

**4. Một chỗ bạn đúng và ReID sai (frame, ID, vì sao):**

`Chưa thể cung cấp frame/ID xác thực vì chưa có model_reid_clip_01.txt, eval_reid_vs_me.json và annotation clip_01. Khi có output, một finding hợp lệ sẽ là frame mà nhãn có bbox xe bốn bánh khớp với gold nhưng ReID bỏ sót hoặc đổi ID; cần ghi cả frame, ID, IoU/ảnh đối chiếu và rule identity áp dụng.`

**5. Một chỗ ReID làm bạn xem lại annotation (frame, ID, vì sao), hoặc lý do evidence cho thấy model sai:**

`Chưa thể cung cấp frame/ID xác thực. Một disagreement chỉ đáng xem lại khi ReID có bbox còn annotation không có, hoặc hai bên cùng có bbox nhưng ID khác. Evidence cho thấy model sai nếu bbox ReID nằm trên vật thể không phải xe bốn bánh, vật thể trong quảng cáo, một track đứng im bất thường, hoặc không khớp gold/ảnh ở frame đó. Không được sửa annotation chỉ vì model khác mình; phải đối chiếu frame và rule occlusion/entry/exit.`

## 6. Nếu phải gán thêm 10 clip nữa

`GUIDELINE_MINI.md cần được điền hoàn chỉnh thay vì để placeholder: ghi ngưỡng occlusion 25 frame, quy tắc cho occlusion dài, ra khỏi khung rồi quay lại mặc định tạo track mới, crossing không đổi ID, bbox chỉ ôm phần nhìn thấy và ngưỡng bắt đầu track khi xe nhỏ/mờ. Mỗi ca mơ hồ phải có clip, frame, ID, quyết định và lý do. Quy trình của tôi sẽ là: làm warm-up; gán clip chính độc lập; tự kiểm ba lượt; peer review có frame-CVAT, frame-MOT, ID, lỗi và closure; chạy validator; khóa pre-gold và manifest hash trước khi xem gold/model; sau đó mới evaluate, rework và chạy ByteTrack/BoT-SORT + ReID. Tôi cũng sẽ lưu thời gian, số track và số keyframe trong lúc làm vì các số này không thể khôi phục từ MOT export.`

## 7. Tệp đã nộp

- [ ] `annotations/clip_01/gt.txt` — chưa có
- [x] `annotations/clip_02/gt.txt`
- [ ] `evidence/pre-gold/clip_01/gt.txt` và `manifest.json` — chưa có
- [ ] `GUIDELINE_MINI.md` đã điền — còn placeholder
- [ ] `outputs/eval_vs_gold.json` — chưa có
- [ ] `outputs/model_bytetrack_clip_01.txt` — chưa có
- [ ] `outputs/model_reid_clip_01.txt` — chưa có
- [ ] `outputs/model_run_config.json` — chưa có
- [ ] `outputs/eval_bytetrack_vs_gold.json`, `outputs/eval_reid_vs_gold.json`, `outputs/eval_reid_vs_me.json` — chưa có
- [ ] `reports/review_partner.md` — chưa có
- [x] `reports/REPORT.md` (file này)
