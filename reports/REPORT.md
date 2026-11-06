# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
`class_id`=  468; 
`class_name`= "cab" ; 
`rank`= 1 ; 
`score`= 0.510915 ; 
`taxonomy_name`= "ImageNet-1K" ;
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Record này sẽ đề cập đến hình bức ảnh
    Record này mô tả kết quả phân loại toàn bộ hình ảnh bằng mô hình YOLO11 Classification. Model phân tích hình ảnh đầu vào và dự đoán hình ảnh thuộc lớp minibus (xe buýt nhỏ), với điểm dự đoán là 0.164284 (khoảng 16,43%). Đây là kết quả phân loại hình ảnh, không phải thông tin về vị trí của một vật thể cụ thể.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách tên lớp (class list) không phải do model tự nghĩ ra mà được xác định từ bộ dữ liệu và cấu hình huấn luyện của model.
    Trong class list này thì thì model sử dụng taxonomy, và các lớp đã được định nghĩa sẵn, Khi huấn luyện, mỗi lớp được gán một ID tương ứng.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Vì class_id giúp xác định lớp bằng mã số, class_name giúp con người dễ đọc và hiểu tên lớp, còn taxonomy_name cho biết lớp đó thuộc bộ danh mục nào. Ghi cả ba giúp dữ liệu rõ ràng, dễ tra cứu và tránh nhầm lẫn.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần quy định rõ cách chọn lớp đại diện cho toàn bộ hình ảnh, chẳng hạn ưu tiên chủ thể chính, đối tượng nổi bật nhất hoặc nội dung chiếm phần lớn hình ảnh. Đồng thời, cần có quy tắc xử lý khi ảnh có nhiều chủ thể để người gán nhãn thống nhất và chỉ chọn một lớp duy nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
    Vì score chỉ thể hiện mức độ tự tin của model đối với kết quả dự đoán, không đảm bảo kết quả đó đúng. Ground truth phải là nhãn thực tế được xác định và kiểm chứng bởi con người hoặc quy trình gán nhãn đáng tin cậy.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: 
`class_name`= "bus" ; 
`score`=  0.912557; 
`bbox_xyxy`=[
      93.17,
      187.95,
      223.01,
      320.91
    ]; 
`bbox_width`=129.84 ; 
`bbox_height`= 132.96;
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Góc trên bên trái ở toạn độ (93.17, 187.95).
    Góc dươi bên phải ở toạn độ (223.01, 320.91).
    Chiều rông = 129.84 pixel.
    Chiều cao = 132.96 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    Ngưỡng 0.35 thì số prediction có score >= 0.35 số vật thể phát hiện là 53.
    Ngưỡng 0.60 thì số prediction có score >= 0.60 số vật thể phát hiện là .
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    Khi hạ ngưỡng,model giữ nhiều hộp dự đoán hơn, trong số model đó sẽ chứa một số hộp sai hoặc không chính xác. Người rivew sẽ phải kiểm tra lại thêm các hộp được dự đoán, xác nhận hộp đúng, loại bỏ hộp sai và chỉnh sửa nhãn hoặc vị trí của box nếu cần.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    Người gán ngãn sẽ phải tạo ra khung chữ nhật bao quang toàn bộ phần vật thể nhìn thấy trong ảnh không gôm gồm quá nhiều vùng nền xung quanh, Khung phải sát với đường viền vật thể nhất có thể, xác định đứng hai góc trên bến trái vf góc dưới bên phải, không được cắt mất phần vật thể nhìn thấy và không gộp nhiều vật thể riêng biệt vào cùng một box. 
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép, sẽ được viết vào Guideline (như phải có quy tắc gán nhãn chung).
    vì Hiện tượng vật thể bị che khuất hoặc bị cắt ở rìa ảnh là cực kỳ phổ biến trong các bộ dữ liệu thực thế
    Làm cho hiện suất dự án đưa trực tiếp hướng dẫn xử lý các ca này vào Guideline kèm ảnh minh họa sẽ giúp toàn bộ đội ngũ gán nhãn thống nhất cách làm ngay lập tức, tiết kiệm thời gian và tránh việc gửi câu hỏi trùng lặp không đáng có lên Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: 
`instance_id`=   "traffic-001"; 
`class_name`=  "bus"; 
`score`=  0.925745; 
số điểm polygon= `polygon_point_count`= 120; 
3–5 điểm đầu của `polygon_xy`=   [[
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    - Box (x_min, y_min, x_max, y_max) só với polygon thì polygon sẽ thêm các chuỗi đỉnh [(x1,y1),...,(xn,yn)] ôm sát với thực thể thay vì chỉ khung chữ nhật bao ngoài, có thể biết được vật nghiên, xoay, cong, khuyết như thế nào còn box thì ko thể hiện được; polygon sẽ tính được diện tích, tỷ lệ che phủ trong box.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - instance_id : dùng đẻ phân biệt nhiều box/polygon trong cùng 1 ảnh, truy vết đối chiếu khi đánh giá, lọc debug biết lỗi thuộc về vật nào của ảnh nào. Tráng trùng lặp id khi gộp nhiều ảnh lại với nhau.
    - Nó **không** phải class ID nghĩa là vật đó thuộc loại gì, và nhiều vật thể khác nhau có thể chung 1 class ID
    - **không** phải tracking ID nghĩa là vật thể đó sẽ vẫn là vật thể đó giữ nguyên số
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    - Chỉ vẽ đúng phần nhìn thấy được của vật, đường polygon ôm sát mép vật 1-2px, khép kín, không lấy nền, không vẽ phần bị che khuất đoán mò.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Nến sử lý bằng guideline 
    - vì đã đúng tye lệ che phủ của từng vật thể, class không gây nhầm không phải ghi sai lable. Còn escalation chỉ zoom lên  vẫn không biết đâu là biên.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)           | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | ------------------------------------------------------  | -------------------------------------------------- | ----------------- | ---------------- |
| Phân loại ảnh         |1 nhãn / ảnh: {sample_id, label}                         | Trong classification_top5.png + classification_predictions.json: top5 conf sát nhau, kiểu kitchen / dining-room khó chọn 1 | Chọn 1 label chính theo guideline, ghi nguồn vào IMAGE_ATTRIBUTION.md | Label có đúng guideline? Top1 có áp đảo không, attribution đủ không? |
| Phát hiện vật thể     |1 dòng / vật: {instance_id: sample-001, class, bbox_xyxy}| Trong detection_predictions.png + .json: person 0.43 lọt trong oven 0.49 + dining table 0.58, conf gần ngưỡng 0.35 → nghi FP| Vẽ box ôm sát phần nhìn thấy, vật chồng thì tách riêng, không vẽ phần bị che đoán mò | Box khít chưa? Có box trùng/nuốt nhau không? Giữ hay xóa FP, check instance_id duy nhất kiểu sample-001 |
| Instance segmentation | 1 dòng / vật: {instance_id, class, bbox + polygon_xy + polygon_point_count} | Trong segmentation_prediction.png + .json: cụm bowl 0.55/0.70/0.74 góc trái-dưới dính nhau, mask dining table 0.58 tràn vào bát, biên mờ| Vẽ polygon khép kín sát mép 1-2px, dừng ở chỗ chạm/che theo guideline  | Polygon sát không lẹm nền? Đã tách chỗ dính đúng chưa? Ca mờ/che >50% thì escalate, chốt PASS đủ 3 JSON + 3 PNG + instance_id duy nhất |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Không phát tán tài liệu ra ngoài và ghi nguồn đẩy đủ. 
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng ngay, không gán nhãn, báo cho reviewer / giảng viên phụ trách lab để quyết định escalate hay loại.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
