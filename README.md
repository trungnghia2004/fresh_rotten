# Hệ Thống Nhận Diện Chất Lượng Trái Cây

Ứng dụng web dùng `FastAPI`, `YOLO` và `CNN` để:
- phát hiện trái cây trong ảnh hoặc camera
- phân loại chất lượng `fresh` / `rotten`

## Thành phần chính

- `app.py`: API chính
- `pipeline_stream.py`: xử lý stream camera/video
- `static/app.js`: giao diện và tracking box phía client
- `weights/cnn_best.keras`: model CNN
- `weights/yolo_fruits_and_vegetables_v3.pt`: model YOLO

Link tải model `YOLO`:
- https://drive.google.com/drive/folders/1I4mtQK11C3p41pO9raR0trgPVj0eQ2yb

## Cài đặt

Khuyến nghị dùng Python `3.10` và môi trường ảo:

```powershell
cd F:\fresh_rotten
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics
```

## Chạy ứng dụng

```powershell
python app.py
```

Hoặc:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt tại:

```txt
http://127.0.0.1:8000
```

## Lưu ý

- Cần có 2 file model trong thư mục `weights`:
  - `cnn_best.keras`
  - `yolo_fruits_and_vegetables_v3.pt`
- Camera trên `localhost` có thể chậm hơn vì máy phải tự đọc webcam, detect, classify và stream lại.
- Nếu box bị lỗi do cache JS cũ, hãy `Ctrl + F5`.

## Ngrok

```powershell
ngrok config add-authtoken <TOKEN_CUA_BAN>
ngrok http 8000
```

## Lỗi thường gặp

- `YOLO detector not loaded`: thiếu `ultralytics` hoặc sai file `.pt`
- `ERR_NGROK_4018`: chưa cấu hình `authtoken` cho `ngrok`
- lỗi dependency `FastAPI/Starlette`: nên cài trong `venv` riêng
