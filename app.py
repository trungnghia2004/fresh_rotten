from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf

try:
    from ultralytics import YOLO
except Exception:  # ultralytics optional
    YOLO = None

# Force Ultralytics to use a local writable config/cache to avoid AppData permission errors
ULTRA_HOME = Path(__file__).resolve().parent / ".ultralytics"
ULTRA_HOME.mkdir(exist_ok=True)
os.environ.setdefault("SETTINGS_DIR", str(ULTRA_HOME))
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", str(ULTRA_HOME))

IMAGE_SIZE: Tuple[int, int] = (224, 224)
TRAIN_DIR = Path(os.getenv("TRAIN_DIR", Path(__file__).resolve().parent / "train"))
CLASS_NAMES_FILE = Path(os.getenv("CLASS_NAMES_FILE", Path(__file__).resolve().parent / "class_names.json"))
ALLOWED_QUALITIES = {"fresh", "rotten"}
DETECT_MODEL = Path(r"F:\fresh_rotten\yolo_fruits_and_vegetables_v3.pt")

DETECT_CONF = float(os.getenv("DETECT_CONF", "0.25"))

MODEL_PATHS = {
    "cnn": Path(__file__).resolve().parent / "cnn_best.keras",
    "mobilenet": Path(__file__).resolve().parent / "mobilenet_best.keras",
}
FALLBACK_FRUITS = [x.strip() for x in os.getenv("FRUIT_NAMES", "").split(",") if x.strip()]

app = FastAPI(title="Fruit Quality Compare")
app.mount("/static", StaticFiles(directory="static"), name="static")

_models: Dict[str, tf.keras.Model] = {}
_class_names: List[str] = []
_label_to_idx: Dict[str, int] = {}
_idx_to_label: Dict[int, str] = {}
_detector: YOLO | None = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"error": f"Internal server error: {exc}"}, status_code=500)


def build_class_names(train_dir: Path) -> List[str]:
    labels: List[str] = []
    if not train_dir.exists():
        return labels

    for fruit_dir in train_dir.iterdir():
        if not fruit_dir.is_dir():
            continue
        fruit = fruit_dir.name.strip()
        for quality_dir in fruit_dir.iterdir():
            if not quality_dir.is_dir():
                continue
            quality = quality_dir.name.strip().lower()
            if quality in ALLOWED_QUALITIES:
                labels.append(f"{fruit}_{quality}")

    return sorted(set(labels))


def load_class_names_from_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    cleaned = []
    for x in data:
        s = str(x).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned


def load_assets() -> None:
    global _models, _class_names, _label_to_idx, _idx_to_label, _detector

    if not _models:
        _class_names = load_class_names_from_file(CLASS_NAMES_FILE)
        if not _class_names:
            _class_names = build_class_names(TRAIN_DIR)
        _label_to_idx = {label: idx for idx, label in enumerate(_class_names)}
        _idx_to_label = {idx: label for label, idx in _label_to_idx.items()}

        for name, path in MODEL_PATHS.items():
            if path.exists():
                _models[name] = tf.keras.models.load_model(path)

        if not _models:
            raise RuntimeError("No model found. Need cnn_best.keras or mobilenet_fruit_quality.keras")

    if _detector is None and YOLO and DETECT_MODEL.exists():
        _detector = YOLO(str(DETECT_MODEL))


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _activation_name(model: tf.keras.Model) -> str | None:
    if not model.layers:
        return None
    activation = getattr(model.layers[-1], "activation", None)
    if activation is None:
        return None
    return getattr(activation, "__name__", None)


def _to_softmax_probs(raw: np.ndarray, activation_name: str | None) -> np.ndarray:
    if activation_name == "softmax":
        probs = raw
    else:
        shifted = raw - np.max(raw)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps)
    probs = np.asarray(probs, dtype=np.float32)
    return probs / np.sum(probs)


def decode_prediction(model: tf.keras.Model, pred: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    output_dim = pred.shape[0]
    class_count = len(_class_names)
    activation_name = _activation_name(model)

    if output_dim > 1:
        probs = _to_softmax_probs(pred, activation_name)
        idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        if class_count == output_dim:
            label = _idx_to_label[idx]
            if "_" in label:
                fruit, quality = label.rsplit("_", 1)
            else:
                fruit, quality = "unknown", label  # binary fresh/rotten labels
        else:
            # Fallback when class mapping from train/ is unavailable:
            # labels were trained as fruit_fresh, fruit_rotten (sorted),
            # so even/odd indices map to fresh/rotten.
            quality = "fresh" if idx % 2 == 0 else "rotten"
            if FALLBACK_FRUITS and len(FALLBACK_FRUITS) * 2 == output_dim:
                fruit = FALLBACK_FRUITS[idx // 2]
                label = f"{fruit}_{quality}"
            else:
                fruit = "unknown"
                label = f"class_{idx}_{quality}"
    elif output_dim == 1 and activation_name == "sigmoid":
        # Binary sigmoid branch: infer positive class from training class map when available.
        score = float(np.clip(pred[0], 0.0, 1.0))
        # Huấn luyện: label 0 = fresh, label 1 = rotten => sigmoid trả xác suất "rotten"
        fruit = "unknown"
        positive_quality = "rotten"  # prob >= thresh -> rotten
        negative_quality = "fresh"   # prob < thresh -> fresh
        thresh = 0.7
        quality = positive_quality if score >= thresh else negative_quality
        confidence = score if score >= thresh else 1.0 - score
        label = f"{fruit}_{quality}"
    else:
        # Fallback for unsupported output shape/activation so API never crashes.
        flat = pred.reshape(-1)
        idx = int(np.argmax(flat))
        confidence = float(np.max(flat))
        label = f"class_{idx}"
        fruit, quality = label, "unknown"

    return {
        "label": label,
        "fruit": fruit,
        "quality": quality,
        "confidence": round(confidence, 4),
    }


def predict_image_models(arr: np.ndarray) -> Dict[str, dict]:
    load_assets()
    results: Dict[str, dict] = {}
    for model_name, model in _models.items():
        pred = model.predict(arr, verbose=0)
        decoded = decode_prediction(model, pred)
        qc = {"fresh": 0, "rotten": 0}
        if decoded["quality"] in qc:
            qc[decoded["quality"]] = 1
        decoded["quality_counts"] = qc
        results[model_name] = decoded
    return results


def aggregate_video_model(model: tf.keras.Model, frames: List[np.ndarray]) -> dict:
    quality_counts = {"fresh": 0, "rotten": 0}
    fruit_counts: Dict[str, int] = {}
    confidence_sum = 0.0

    for arr in frames:
        pred = model.predict(arr, verbose=0)
        item = decode_prediction(model, pred)
        quality = item["quality"]
        fruit = item["fruit"]
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        fruit_counts[fruit] = fruit_counts.get(fruit, 0) + 1
        confidence_sum += float(item["confidence"])

    sampled_frames = len(frames)
    if sampled_frames == 0:
        raise ValueError("No sampled frame")

    final_quality = "fresh" if quality_counts["fresh"] >= quality_counts["rotten"] else "rotten"
    final_fruit = max(fruit_counts, key=fruit_counts.get)

    return {
        "label": f"{final_fruit}_{final_quality}",
        "fruit": final_fruit,
        "quality": final_quality,
        "confidence": round(confidence_sum / sampled_frames, 4),
        "quality_counts": quality_counts,
        "sampled_frames": sampled_frames,
    }


def detect_and_crop(img: Image.Image):
    """
    Run YOLO once (using its internal default conf), return boxes sorted by confidence (desc).
    """
    if _detector is None:
        return [], 0.0, {"attempts": []}

    img_rgb = img.convert("RGB")
    res = _detector(img_rgb, imgsz=640, verbose=False)[0]
    names = _detector.model.names  # type: ignore

    outputs = []
    debug_list = []
    if len(res.boxes):
        sorted_idx = np.argsort(res.boxes.conf.cpu().numpy())[::-1]
        W, H = img.size
        pad_ratio = 0.05  # mở rộng 5% mỗi chiều trước khi crop
        for idx in sorted_idx:
            box = res.boxes.xyxy[idx]
            cls = res.boxes.cls[idx]
            conf = float(res.boxes.conf[idx])
            cls_name = names.get(int(cls), str(int(cls)))
            x1, y1, x2, y2 = map(float, box.cpu().numpy().tolist())
            # mở rộng box
            w = x2 - x1
            h = y2 - y1
            dx = w * pad_ratio
            dy = h * pad_ratio
            x1e = max(0, x1 - dx)
            y1e = max(0, y1 - dy)
            x2e = min(W, x2 + dx)
            y2e = min(H, y2 + dy)
            crop = img.crop((x1e, y1e, x2e, y2e)).convert("RGB")
            outputs.append(
                {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "det_label": cls_name,
                    "det_confidence": conf,
                    "crop": crop,
                }
            )
            debug_list.append({"label": cls_name, "conf": conf, "box": [int(x1), int(y1), int(x2), int(y2)], "expanded_box": [int(x1e), int(y1e), int(x2e), int(y2e)]})

    max_conf = float(res.boxes.conf.max().item()) if len(res.boxes.conf) else 0.0
    dbg = [{"imgsz": 640, "conf": "default", "max_conf": max_conf, "boxes": len(res.boxes), "tops": debug_list}]
    return outputs, max_conf, {"attempts": dbg}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Please upload an image."}, status_code=400)

    try:
        load_assets()
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        if _detector is None:
            return JSONResponse({"error": "YOLO detector not loaded. Ensure ultralytics installed and model path exists."}, status_code=500)

        det_list, max_conf, dbg = detect_and_crop(img)
        if not det_list:
            return JSONResponse({"error": f"No fruits detected by YOLO. max_conf={max_conf:.3f}", "debug": dbg}, status_code=400)

        # Lấy nhãn có confidence cao nhất, chỉ giữ các box cùng nhãn và conf >= 0.6
        top_label = det_list[0]["det_label"]
        filtered = [d for d in det_list if d["det_label"] == top_label and d["det_confidence"] >= 0.6]
        if not filtered:
            filtered = [det_list[0]]

        detections = []
        fruit_counts = {}
        crop_count = len(filtered)
        total_qc = {"fresh": 0, "rotten": 0}
        for det in filtered:
            fruit = det["det_label"]
            # Chuẩn bị ảnh 224x224 cho model
            arr = preprocess_image(det["crop"])
            models = predict_image_models(arr)
            for m in models.values():
                m["fruit"] = fruit
                m["label"] = f"{fruit}_{m['quality']}"
                m["crop_count"] = crop_count
            agg_model = models.get("mobilenet") or next(iter(models.values()))
            quality = agg_model["quality"]
            if quality in total_qc:
                total_qc[quality] += 1
            fc = fruit_counts.setdefault(fruit, {"total": 0, "fresh": 0, "rotten": 0})
            fc["total"] += 1
            if quality in fc:
                fc[quality] += 1
            detections.append(
                {
                    "detection": {
                        "box": det["box"],
                        "label": fruit,
                        "confidence": det["det_confidence"],
                    },
                    "models": models,
                }
            )

        main_detection = max(detections, key=lambda d: d["detection"]["confidence"], default=None)
        # gán tổng số fresh/rotten và crop_count vào models của main_detection để UI hiển thị
        if main_detection and "models" in main_detection:
            for m in main_detection["models"].values():
                m["quality_counts"] = total_qc
                m["crop_count"] = crop_count

        return {
            "mode": "image",
            "detections": detections,
            "fruit_counts": fruit_counts,
            "sampled_frames": 1,
            "main_detection": main_detection,
            "debug": dbg,
        }
    except Exception as exc:
        return JSONResponse({"error": f"Image prediction failed: {exc}"}, status_code=500)


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        return JSONResponse({"error": "Please upload a video."}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        load_assets()
        import cv2

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Cannot read video."}, status_code=400)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(frame_count // 12, 1)

        sampled: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                sampled.append(preprocess_image(img))
            idx += 1

        cap.release()

        if not sampled:
            return JSONResponse({"error": "No frame sampled from video."}, status_code=400)

        load_assets()
        models = {name: aggregate_video_model(model, sampled) for name, model in _models.items()}
        return {
            "mode": "video",
            "models": models,
            "sampled_frames": len(sampled),
        }
    except Exception as exc:
        return JSONResponse({"error": f"Video prediction failed: {exc}"}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
