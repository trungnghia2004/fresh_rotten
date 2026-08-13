# Hệ Thống Nhận Diện Chất Lượng Trái Cây

Ứng dụng web dùng `FastAPI` để nhận diện trái cây và phân loại chất lượng `fresh` / `rotten`.

Hệ thống hiện tại sử dụng:
- `YOLO` để phát hiện vị trí trái cây trong ảnh hoặc khung hình camera
- `CNN` (`cnn_best.keras`) để phân loại chất lượng của từng trái cây đã được cắt ra từ vùng phát hiện
- giao diện web đơn giản để chạy với ảnh, camera và luồng video nội bộ

## Chức năng chính

- Tải ảnh lên để nhận diện và phân loại chất lượng trái cây
- Chạy camera trực tiếp trên trình duyệt
- Stream camera từ máy chạy server
- Vẽ box quanh trái cây và hiển thị nhãn chất lượng
- Tối ưu camera theo hướng:
  - detect thưa hơn
  - tracking dày hơn
  - classify không chạy ở mọi lần detect để giảm độ trễ

## Cấu trúc quan trọng

- `app.py`: API chính, xử lý ảnh, stream video/camera, load model
- `pipeline_stream.py`: luồng xử lý stream camera/video phía server
- `static/app.js`: logic giao diện, camera trình duyệt, tracking box phía client
- `templates/index.html`: giao diện HTML
- `weights/cnn_best.keras`: model CNN phân loại chất lượng
- `weights/yolo_fruits_and_vegetables_v3.pt`: model YOLO phát hiện trái cây

## Yêu cầu môi trường

- Python `3.10` được khuyến nghị
- Windows là môi trường đã được dùng để chạy và chỉnh sửa dự án này
- Nếu muốn dùng đầy đủ tính năng detect/camera, cần có `ultralytics`

## Cài đặt

Nên dùng môi trường ảo riêng để tránh xung đột dependency.

```powershell
cd F:\fresh_rotten
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chưa có `ultralytics`, cài thêm:

```powershell
pip install ultralytics
```

Nếu máy có GPU NVIDIA và muốn tận dụng GPU cho `YOLO` hoặc `TensorFlow`, cần cài đúng stack CUDA tương ứng. Nếu không, ứng dụng vẫn chạy bằng CPU.

## Kiểm tra file model

Trước khi chạy, bảo đảm các file sau tồn tại:

- `weights/cnn_best.keras`
- `weights/yolo_fruits_and_vegetables_v3.pt`

Nếu thiếu file `YOLO`, API sẽ báo lỗi kiểu:

```txt
YOLO detector not loaded. Ensure ultralytics installed and model path exists.
```

## Chạy ứng dụng

Có 2 cách chạy.

Cách 1:

```powershell
python app.py
```

Cách 2:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Sau khi chạy, mở:

```txt
http://127.0.0.1:8000
```

## Cách hoạt động

### 1. Ảnh

- người dùng tải ảnh lên
- server chạy `YOLO` để phát hiện trái cây
- từng vùng ảnh được crop ra
- `CNN` phân loại `fresh` / `rotten`
- server trả kết quả box + nhãn cho frontend

### 2. Camera trình duyệt

- trình duyệt lấy camera bằng `getUserMedia`
- frontend gửi frame lên API theo chu kỳ
- không phải mọi lần detect đều chạy classify
- box được tracking liên tục ở phía client để giảm cảm giác trễ

### 3. Camera phía server

- khi mở bằng `localhost`, ứng dụng ưu tiên stream camera từ máy chạy server
- luồng này thường nặng hơn camera trình duyệt vì máy phải:
  - đọc webcam
  - chạy `YOLO`
  - định kỳ chạy `CNN`
  - encode stream trả về browser

Vì vậy, khi chạy trực tiếp trên laptop, FPS có thể tụt rõ hơn so với khi dùng camera phía client.

## Một số biến môi trường quan trọng

Dự án có khá nhiều biến môi trường để tinh chỉnh hiệu năng. Một số biến đáng chú ý trong `app.py`:

- `DETECT_CONF`: ngưỡng confidence của `YOLO`
- `STREAM_YOLO_IMGSZ`: kích thước suy luận cho stream video
- `CAMERA_STREAM_YOLO_IMGSZ`: kích thước suy luận cho camera server-side
- `CAMERA_CAPTURE_WIDTH`, `CAMERA_CAPTURE_HEIGHT`: độ phân giải camera server-side
- `CAMERA_STREAM_TARGET_FPS`: FPS mục tiêu của camera server-side
- `USE_TF_GPU`: bật/tắt GPU cho TensorFlow

Nếu muốn giảm tải máy khi chạy local camera, nên giảm:

- `CAMERA_STREAM_YOLO_IMGSZ`
- `CAMERA_CAPTURE_WIDTH`
- `CAMERA_CAPTURE_HEIGHT`
- `CAMERA_STREAM_TARGET_FPS`

## Gợi ý tối ưu hiệu năng

### Khi box bị chậm hoặc camera không mượt

Ưu tiên:
- giảm độ phân giải frame gửi đi
- không classify ở mọi lần detect
- tracking box phía client hoặc phía server
- giảm số box tối đa cần xử lý

### Khi chạy trực tiếp trên laptop

Nên thử:
- dùng camera trình duyệt thay vì camera server-side
- đóng bớt ứng dụng khác đang dùng CPU/GPU
- cắm sạc, tắt chế độ tiết kiệm pin
- giảm `imgsz` của YOLO

## Dùng với ngrok

Nếu muốn chia sẻ ra ngoài mạng nội bộ:

```powershell
ngrok config add-authtoken <TOKEN_CUA_BAN>
ngrok http 8000
```

Lưu ý:
- domain miễn phí của `ngrok` thường được cấp ngẫu nhiên
- không thể tự chọn tên domain đẹp ở gói free

## Lỗi thường gặp

### 1. `ERR_NGROK_4018`

Nguyên nhân: chưa đăng nhập `ngrok` hoặc chưa cấu hình `authtoken`.

### 2. `YOLO detector not loaded`

Nguyên nhân thường là:
- chưa cài `ultralytics`
- sai đường dẫn file `.pt`
- model không tồn tại trong thư mục `weights`

### 3. Lỗi dependency `FastAPI` / `Starlette`

Nên dùng môi trường ảo riêng và cài đúng theo `requirements.txt`.

### 4. Camera bị ngược hoặc box rung

Phần này đã được tối ưu trong frontend, nhưng nếu browser đang cache JS cũ thì cần:

```txt
Ctrl + F5
```

hoặc đóng tab và mở lại.
