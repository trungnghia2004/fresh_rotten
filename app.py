from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pipeline_stream import StreamConfig, stream_camera_frames, stream_video_frames

try:
    from ultralytics import YOLO
except Exception:  # ultralytics optional
    YOLO = None

# Force Ultralytics to use a local writable config/cache to avoid AppData permission errors
ULTRA_HOME = Path(__file__).resolve().parent / ".ultralytics"
ULTRA_HOME.mkdir(exist_ok=True)
os.environ.setdefault("SETTINGS_DIR", str(ULTRA_HOME))
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", str(ULTRA_HOME))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

IMAGE_SIZE: Tuple[int, int] = (224, 224)
TRAIN_DIR = Path(os.getenv("TRAIN_DIR", BASE_DIR / "train"))
CLASS_NAMES_FILE = Path(os.getenv("CLASS_NAMES_FILE", BASE_DIR / "class_names.json"))
ALLOWED_QUALITIES = {"fresh", "rotten"}
DETECT_MODEL = Path(r"F:\group23_22001611_22001624\weights\yolo_fruits_and_vegetables_v3.pt")

DETECT_CONF = float(os.getenv("DETECT_CONF", "0.70"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.2"))
DETECT_MAX_DET = max(1, int(os.getenv("DETECT_MAX_DET", "100")))
DETECT_MIN_AREA_RATIO = float(os.getenv("DETECT_MIN_AREA_RATIO", "0.005"))
DETECT_ALLOWED_LABELS = {
    x.strip().lower() for x in os.getenv("DETECT_ALLOWED_LABELS", "").split(",") if x.strip()
}
USE_TF_GPU = os.getenv("USE_TF_GPU", "0") == "1"
VIDEO_FRAME_STEP = max(1, int(os.getenv("VIDEO_FRAME_STEP", "1")))
VIDEO_MAX_BOXES = max(1, int(os.getenv("VIDEO_MAX_BOXES", "3")))
STREAM_DETECT_EVERY = max(1, int(os.getenv("STREAM_DETECT_EVERY", "4")))
STREAM_CLASSIFY_EVERY = max(1, int(os.getenv("STREAM_CLASSIFY_EVERY", "3")))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "65"))
STREAM_YOLO_IMGSZ = max(320, int(os.getenv("STREAM_YOLO_IMGSZ", "512")))
STREAM_TARGET_FPS = float(os.getenv("STREAM_TARGET_FPS", "30"))
STREAM_OUTPUT_MAX_WIDTH = max(0, int(os.getenv("STREAM_OUTPUT_MAX_WIDTH", "720")))
# Đồng bộ ngưỡng detect cho toàn bộ ảnh/video/camera.
# Có thể override riêng bằng STREAM_MIN_CONF nếu cần, nhưng mặc định theo DETECT_CONF.
STREAM_MIN_CONF = float(os.getenv("STREAM_MIN_CONF", str(DETECT_CONF)))
STREAM_MAX_BOXES = max(1, int(os.getenv("STREAM_MAX_BOXES", "10")))
STREAM_CLASSIFY_MAX_BOXES = max(1, int(os.getenv("STREAM_CLASSIFY_MAX_BOXES", "10")))
ANNOTATION_FONT_SIZE = max(10, int(os.getenv("ANNOTATION_FONT_SIZE", "18")))
ANNOTATION_MAX_WIDTH = max(0, int(os.getenv("ANNOTATION_MAX_WIDTH", "960")))
CAMERA_STREAM_TARGET_FPS = float(os.getenv("CAMERA_STREAM_TARGET_FPS", "30"))
CAMERA_STREAM_DETECT_EVERY = max(1, int(os.getenv("CAMERA_STREAM_DETECT_EVERY", "1")))
CAMERA_STREAM_CLASSIFY_EVERY = max(1, int(os.getenv("CAMERA_STREAM_CLASSIFY_EVERY", "1")))
CAMERA_STREAM_JPEG_QUALITY = int(os.getenv("CAMERA_STREAM_JPEG_QUALITY", "88"))
CAMERA_STREAM_YOLO_IMGSZ = max(320, int(os.getenv("CAMERA_STREAM_YOLO_IMGSZ", "640")))
CAMERA_STREAM_OUTPUT_MAX_WIDTH = max(0, int(os.getenv("CAMERA_STREAM_OUTPUT_MAX_WIDTH", "960")))
CAMERA_STREAM_MAX_BOXES = max(1, int(os.getenv("CAMERA_STREAM_MAX_BOXES", "10")))
CAMERA_STREAM_CLASSIFY_MAX_BOXES = max(1, int(os.getenv("CAMERA_STREAM_CLASSIFY_MAX_BOXES", "10")))
CAMERA_CAPTURE_WIDTH = max(0, int(os.getenv("CAMERA_CAPTURE_WIDTH", "1280")))
CAMERA_CAPTURE_HEIGHT = max(0, int(os.getenv("CAMERA_CAPTURE_HEIGHT", "720")))
CAMERA_CAPTURE_FPS = float(os.getenv("CAMERA_CAPTURE_FPS", "30"))
CAMERA_CAPTURE_BUFFER_SIZE = max(0, int(os.getenv("CAMERA_CAPTURE_BUFFER_SIZE", "1")))
CAMERA_ASYNC_INFERENCE = os.getenv("CAMERA_ASYNC_INFERENCE", "1") == "1"
CAMERA_LITE_MAX_BOXES = max(1, int(os.getenv("CAMERA_LITE_MAX_BOXES", "3")))

MODEL_PATHS = {
    "cnn": BASE_DIR / "weights/cnn_best.keras",
    "mobilenet": BASE_DIR / "weights/mobilenet_fruit_quality.keras",
}
FALLBACK_FRUITS = [x.strip() for x in os.getenv("FRUIT_NAMES", "").split(",") if x.strip()]

app = FastAPI(title="Fruit Quality Compare")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_models: Dict[str, tf.keras.Model] = {}
_class_names: List[str] = []
_label_to_idx: Dict[str, int] = {}
_idx_to_label: Dict[int, str] = {}
_detector: YOLO | None = None
_stream_jobs: Dict[str, str] = {}
_camera_jobs: Dict[str, int] = {}
try:
    _font = ImageFont.truetype("arial.ttf", ANNOTATION_FONT_SIZE)
except Exception:
    _font = ImageFont.load_default()
_custom_objects = {}
tf = None
_patched_input_layer = None
_patched_batch_norm = None


def _get_tf():
    global tf, _patched_input_layer, _patched_batch_norm
    if tf is None:
        import tensorflow as _tf

        tf = _tf

        class PatchedInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, *args, **kwargs):
                if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                    kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
                super().__init__(*args, **kwargs)

        class PatchedBatchNorm(tf.keras.layers.BatchNormalization):
            def __init__(self, *args, **kwargs):
                kwargs.pop("synchronized", None)
                super().__init__(*args, **kwargs)

        _patched_input_layer = PatchedInputLayer
        _patched_batch_norm = PatchedBatchNorm
    return tf


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"error": f"Internal server error: {exc}"}, status_code=500)

@app.on_event("startup")
async def on_startup():
    try:
        load_assets()
        print("[INIT] Models loaded.")
    except Exception as exc:
        print(f"[INIT] Warmup failed: {exc}")



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
        tf_mod = _get_tf()
        if not USE_TF_GPU:
            try:
                tf_mod.config.set_visible_devices([], "GPU")
            except Exception:
                pass

        _class_names = load_class_names_from_file(CLASS_NAMES_FILE)
        if not _class_names:
            _class_names = build_class_names(TRAIN_DIR)
        _label_to_idx = {label: idx for idx, label in enumerate(_class_names)}
        _idx_to_label = {idx: label for label, idx in _label_to_idx.items()}

        for name, path in MODEL_PATHS.items():
            if path.exists():
                _models[name] = tf_mod.keras.models.load_model(
                    path,
                    compile=False,
                    custom_objects={
                        "InputLayer": _patched_input_layer,
                        "DTypePolicy": tf_mod.keras.mixed_precision.Policy,
                        "BatchNormalization": _patched_batch_norm,
                    },
                )

        if not _models:
            raise RuntimeError("No model found. Need cnn_best.keras or mobilenet_fruit_quality.keras")

    if _detector is None and YOLO and DETECT_MODEL.exists():
        _detector = YOLO(str(DETECT_MODEL))
        try:
            _detector.to("cuda")
        except Exception:
            pass



def ensure_detector() -> bool:
    global _detector
    if _detector is not None:
        return True
    if not YOLO or not DETECT_MODEL.exists():
        return False
    try:
        _detector = YOLO(str(DETECT_MODEL))
        try:
            _detector.to("cuda")
        except Exception:
            pass
        return True
    except Exception:
        return False


def _open_cv_camera(source: int):
    import cv2

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    if cap.isOpened():
        if CAMERA_CAPTURE_WIDTH > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_CAPTURE_WIDTH))
        if CAMERA_CAPTURE_HEIGHT > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_CAPTURE_HEIGHT))
        if CAMERA_CAPTURE_FPS > 0:
            cap.set(cv2.CAP_PROP_FPS, float(CAMERA_CAPTURE_FPS))
        if CAMERA_CAPTURE_BUFFER_SIZE > 0:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, float(CAMERA_CAPTURE_BUFFER_SIZE))
            except Exception:
                pass
    return cap


def _probe_camera_source(source: int, samples: int = 8):
    import cv2

    cap = _open_cv_camera(source)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None

    score_sum = 0.0
    ok_frames = 0
    try:
        for _ in range(max(1, samples)):
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Chấm điểm dựa trên độ tương phản + độ sắc cạnh.
            std_gray = float(np.std(gray))
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            score = std_gray + min(lap_var, 400.0) * 0.1
            score_sum += score
            ok_frames += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if ok_frames == 0:
        return None
    return {
        "source": int(source),
        "score": score_sum / float(ok_frames),
        "ok_frames": ok_frames,
    }


def _pick_best_camera_source(candidates: List[int] | None = None):
    if not candidates:
        candidates = [0, 1, 2]

    best = None
    for src in candidates:
        info = _probe_camera_source(int(src))
        if info is None:
            continue
        if best is None or float(info["score"]) > float(best["score"]):
            best = info

    if best is None:
        return None, {"candidates": candidates, "probes": []}

    return int(best["source"]), best


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_batch(crops: List[Image.Image]) -> np.ndarray:
    if not crops:
        return np.empty((0, *IMAGE_SIZE, 3), dtype=np.float32)
    arrs = []
    for im in crops:
        im = im.convert("RGB").resize(IMAGE_SIZE)
        arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(arrs, axis=0)


def pil_to_data_url(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    import base64
    import io

    buf = io.BytesIO()
    save_kwargs = {}
    fmt_u = (fmt or "PNG").upper()
    if fmt_u == "JPEG":
        save_kwargs["quality"] = int(max(40, min(95, quality)))
        save_kwargs["optimize"] = True
    img.convert("RGB").save(buf, format=fmt_u, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt_u == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"

def render_annotated(
    image: Image.Image,
    detections: List[dict],
    model_name: str = "mobilenet",
    fmt: str = "PNG",
    quality: int = 85,
) -> str:
    """Vẽ box + nhãn fruit_quality lên ảnh, trả về data URL base64."""
    if not detections:
        return ""
    img, draw_detections = resize_pil_for_annotation(image, detections, ANNOTATION_MAX_WIDTH)
    img = annotate_pil(img, draw_detections, model_name=model_name)
    return pil_to_data_url(img, fmt=fmt, quality=quality)


def scale_detections(detections: List[dict], sx: float, sy: float) -> List[dict]:
    if sx == 1.0 and sy == 1.0:
        return detections

    scaled = []
    for det in detections:
        copied = dict(det)
        copied_detection = dict(copied.get("detection", {}))
        box = copied_detection.get("box")
        if box:
            x1, y1, x2, y2 = box
            copied_detection["box"] = [
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
            ]
        copied["detection"] = copied_detection
        scaled.append(copied)
    return scaled


def resize_pil_for_annotation(image: Image.Image, detections: List[dict], max_width: int):
    if max_width <= 0 or image.width == max_width:
        return image.convert("RGB").copy(), detections

    scale = max_width / float(image.width)
    out_h = max(2, int(round(image.height * scale)))
    resized = image.convert("RGB").resize((max_width, out_h), Image.Resampling.LANCZOS)
    return resized, scale_detections(detections, scale, scale)


def annotate_pil(image: Image.Image, detections: List[dict], model_name: str = "mobilenet") -> Image.Image:
    """Vẽ box + nhãn fruit_quality lên ảnh, trả về ảnh PIL."""
    img = image.convert("RGB").copy()
    if not detections:
        return img
    draw = ImageDraw.Draw(img)
    for det in detections:
        box = det["detection"]["box"]
        fruit = det["detection"]["label"]
        quality = None
        if det.get("models"):
            if model_name in det["models"]:
                quality = det["models"][model_name].get("quality")
            if quality is None and det["models"]:
                first_model = next(iter(det["models"].values()))
                quality = first_model.get("quality")
        label = f"{fruit}_{quality}" if quality else fruit
        x1, y1, x2, y2 = box
        line_width = 2
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=line_width)
        text = label
        try:
            bbox = draw.textbbox((0, 0), text, font=_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = _font.getsize(text)
        label_y = max(0, y1 - th - 6)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=(0, 0, 0))
        draw.text((x1 + 4, label_y + 2), text, fill=(255, 255, 255), font=_font)
    return img

def annotate_cv2(frame_bgr: np.ndarray, detections: List[dict]) -> np.ndarray:
    import cv2

    out = frame_bgr.copy()
    if not detections:
        return out

    for det in detections:
        x1, y1, x2, y2 = det["detection"]["box"]
        fruit = det["detection"]["label"]
        quality = None
        if det.get("models"):
            if "mobilenet" in det["models"]:
                quality = det["models"]["mobilenet"].get("quality")
            if quality is None and det["models"]:
                first_model = next(iter(det["models"].values()))
                quality = first_model.get("quality")
        label = f"{fruit}_{quality}" if quality else fruit

        font_scale = 0.5
        thickness = 1
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 8, ty + th + baseline + 5), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 4, ty + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def box_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _box_area(box: List[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_intersection(a: List[int], b: List[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _box_overlap_min_ratio(a: List[int], b: List[int]) -> float:
    inter = _box_intersection(a, b)
    if inter <= 0:
        return 0.0
    amin = min(_box_area(a), _box_area(b))
    if amin <= 0:
        return 0.0
    return inter / float(amin)


def deduplicate_detections(
    items: List[dict],
    *,
    iou_thresh: float = 0.45,
    overlap_min_thresh: float = 0.75,
) -> tuple[List[dict], int]:
    """
    Remove overlapping duplicate detections of the same class.
    Keep the highest-confidence item when two boxes are near-identical/contained.
    """
    if not items:
        return items, 0

    ordered = sorted(items, key=lambda x: float(x.get("det_confidence", 0.0)), reverse=True)
    kept: List[dict] = []
    removed = 0

    for cand in ordered:
        cbox = cand.get("box")
        clabel = str(cand.get("det_label", ""))
        if not cbox or len(cbox) != 4:
            continue

        duplicate = False
        for k in kept:
            kbox = k.get("box")
            if not kbox or len(kbox) != 4:
                continue
            if str(k.get("det_label", "")) != clabel:
                continue

            if box_iou(cbox, kbox) >= iou_thresh or _box_overlap_min_ratio(cbox, kbox) >= overlap_min_thresh:
                duplicate = True
                removed += 1
                break

        if not duplicate:
            kept.append(cand)

    return kept, removed


def attach_quality_from_previous(current: List[dict], previous: List[dict]) -> None:
    if not current or not previous:
        return

    for cur in current:
        if cur.get("models"):
            continue
        cur_label = cur.get("detection", {}).get("label")
        cur_box = cur.get("detection", {}).get("box")
        if not cur_label or not cur_box:
            continue

        best = None
        best_iou = 0.0
        for prev in previous:
            prev_label = prev.get("detection", {}).get("label")
            prev_box = prev.get("detection", {}).get("box")
            if prev_label != cur_label or not prev_box:
                continue
            iou = box_iou(cur_box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best = prev

        if best is not None and best_iou >= 0.3:
            prev_models = best.get("models") or {}
            mb = prev_models.get("mobilenet")
            if mb:
                cur["models"] = {
                    "mobilenet": {
                        "fruit": mb.get("fruit", cur_label),
                        "quality": mb.get("quality", "unknown"),
                        "confidence": mb.get("confidence", 0.0),
                        "label": f"{cur_label}_{mb.get('quality', 'unknown')}",
                        "quality_counts": mb.get("quality_counts", {"fresh": 0, "rotten": 0}),
                    }
                }


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


def predict_models_batch(arr_batch: np.ndarray, model_names: List[str] | None = None) -> List[Dict[str, dict]]:
    """
    Chạy cả hai model trên batch crop một lần để giảm thời gian.
    Trả về list cùng độ dài batch, mỗi phần tử chứa kết quả theo model.
    """
    load_assets()
    if arr_batch.shape[0] == 0:
        return []
    batch_size = arr_batch.shape[0]
    # chuẩn bị kết quả rỗng
    out: List[Dict[str, dict]] = [dict() for _ in range(batch_size)]

    def _predict_one(name_model):
        name, model = name_model
        preds = model.predict(arr_batch, verbose=0)
        return name, model, preds

    if model_names:
        wanted = set(model_names)
        items = [(name, model) for name, model in _models.items() if name in wanted]
    else:
        items = list(_models.items())
    if not items:
        return out
    # chạy song song CNN + MobileNet trên cùng batch crop
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(2, len(items)))) as ex:
            futures = [ex.submit(_predict_one, nm) for nm in items]
            for fut in as_completed(futures):
                name, model, preds = fut.result()
                for i in range(batch_size):
                    decoded = decode_prediction(model, preds[i])
                    qc = {"fresh": 0, "rotten": 0}
                    if decoded["quality"] in qc:
                        qc[decoded["quality"]] = 1
                    decoded["quality_counts"] = qc
                    out[i][name] = decoded
    except Exception:
        # fallback an toàn nếu runtime không thread-safe
        for name, model in items:
            preds = model.predict(arr_batch, verbose=0)
            for i in range(batch_size):
                decoded = decode_prediction(model, preds[i])
                qc = {"fresh": 0, "rotten": 0}
                if decoded["quality"] in qc:
                    qc[decoded["quality"]] = 1
                decoded["quality_counts"] = qc
                out[i][name] = decoded
    return out


def detect_and_crop(
    img: Image.Image,
    imgsz: int = 640,
    conf: float | None = None,
    collect_debug: bool = True,
):
    """
    Run YOLO with explicit conf/iou filters and return boxes sorted by confidence (desc).
    """
    if _detector is None:
        return [], 0.0, {"attempts": []}

    img_rgb = img.convert("RGB")
    res = _detector(
        img_rgb,
        imgsz=imgsz,
        conf=DETECT_CONF if conf is None else conf,
        iou=DETECT_IOU,
        max_det=DETECT_MAX_DET,
        agnostic_nms=True,
        verbose=False,
    )[0]
    names = _detector.model.names  # type: ignore

    outputs = []
    debug_list: List[dict] = []
    filtered_small = 0
    filtered_label = 0
    if len(res.boxes):
        sorted_idx = np.argsort(res.boxes.conf.cpu().numpy())[::-1]
        W, H = img.size
        total_area = float(max(1, W * H))
        pad_ratio = 0.05  # mở rộng 5% mỗi chiều trước khi crop
        for idx in sorted_idx:
            box = res.boxes.xyxy[idx]
            cls = res.boxes.cls[idx]
            conf = float(res.boxes.conf[idx])
            cls_name = names.get(int(cls), str(int(cls)))
            x1, y1, x2, y2 = map(float, box.cpu().numpy().tolist())

            box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / total_area)
            if box_area_ratio < DETECT_MIN_AREA_RATIO:
                filtered_small += 1
                continue
            if DETECT_ALLOWED_LABELS and cls_name.strip().lower() not in DETECT_ALLOWED_LABELS:
                filtered_label += 1
                continue

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
                    "det_area_ratio": round(box_area_ratio, 6),
                    "crop": crop,
                }
            )
            if collect_debug:
                debug_list.append(
                    {
                        "label": cls_name,
                        "conf": conf,
                        "area_ratio": round(box_area_ratio, 6),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "expanded_box": [int(x1e), int(y1e), int(x2e), int(y2e)],
                    }
                )

    max_conf = float(res.boxes.conf.max().item()) if len(res.boxes.conf) else 0.0
    dbg = []
    if collect_debug:
        dbg = [
            {
                "imgsz": imgsz,
                "conf": DETECT_CONF if conf is None else conf,
                "iou": DETECT_IOU,
                "max_conf": max_conf,
                "boxes_raw": len(res.boxes),
                "boxes_kept": len(outputs),
                "filtered_small": filtered_small,
                "filtered_label": filtered_label,
                "tops": debug_list,
            }
        ]
    return outputs, max_conf, {"attempts": dbg}
@app.get("/", response_class=HTMLResponse)
def index():
    template_path = TEMPLATES_DIR / "index.html"
    if not template_path.exists():
        return HTMLResponse("<h1>Missing template: templates/index.html</h1>", status_code=500)
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict_image")
async def predict_image(
    file: UploadFile = File(...),
    selected_model: str | None = Form(None),
    allow_no_detection: str | None = Form(None),
    imgsz: int | None = Form(None),
    lite: str | None = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Please upload an image."}, status_code=400)

    try:
        load_assets()
        selected = (selected_model or "").strip().lower() if selected_model else ""
        if selected and selected not in _models:
            return JSONResponse({"error": "selected_model must be 'cnn' or 'mobilenet'."}, status_code=400)

        lite_mode = (lite or "").strip() == "1"
        render_model_names = [selected] if selected else list(_models.keys())
        req_imgsz = 640
        if imgsz is not None:
            req_imgsz = max(320, min(960, int(imgsz)))

        data = await file.read()
        img = Image.open(io.BytesIO(data))
        if _detector is None:
            return JSONResponse({"error": "YOLO detector not loaded. Ensure ultralytics installed and model path exists."}, status_code=500)

        det_list, max_conf, dbg = detect_and_crop(img, imgsz=req_imgsz, collect_debug=not lite_mode)
        if not det_list:
            if (allow_no_detection or "").strip() == "1":
                raw_img = pil_to_data_url(img)
                payload = {
                    "mode": "image",
                    "detections": [],
                    "fruit_counts": {},
                    "sampled_frames": 1,
                    "main_detection": None,
                    "annotated_image": raw_img,
                    "notice": f"No fruits detected by YOLO. max_conf={max_conf:.3f}",
                }
                if not lite_mode:
                    payload["annotated_images"] = {name: raw_img for name in render_model_names}
                    payload["debug"] = dbg
                return payload
            return JSONResponse({"error": f"No fruits detected by YOLO. max_conf={max_conf:.3f}", "debug": dbg}, status_code=400)

        filtered = [d for d in det_list if d["det_confidence"] >= DETECT_CONF]
        if not filtered:
            filtered = det_list
        if lite_mode:
            filtered = filtered[:CAMERA_LITE_MAX_BOXES]

        detections = []
        fruit_counts = {}
        crop_count = len(filtered)
        total_qc_by_model = {name: {"fresh": 0, "rotten": 0} for name in render_model_names}

        crops = [d["crop"] for d in filtered]
        arr_batch = preprocess_batch(crops)
        model_names = [selected] if selected else None
        models_batch = predict_models_batch(arr_batch, model_names=model_names)

        for det, models in zip(filtered, models_batch):
            fruit = det["det_label"]
            for m in models.values():
                m["fruit"] = fruit
                m["label"] = f"{fruit}_{m['quality']}"
                m["crop_count"] = crop_count
            for model_name, model_result in models.items():
                quality = model_result.get("quality")
                if model_name in total_qc_by_model and quality in total_qc_by_model[model_name]:
                    total_qc_by_model[model_name][quality] += 1
            agg_model = models.get("mobilenet") or next(iter(models.values()))
            quality = agg_model["quality"]
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
            for name, m in main_detection["models"].items():
                m["quality_counts"] = total_qc_by_model.get(name, {"fresh": 0, "rotten": 0})
                m["crop_count"] = crop_count

        render_fmt = "JPEG" if lite_mode else "PNG"
        render_quality = 80 if lite_mode else 90
        annotated_images = {
            name: render_annotated(img, detections, model_name=name, fmt=render_fmt, quality=render_quality)
            for name in render_model_names
        }
        if selected:
            annotated_image = annotated_images.get(selected, "")
        else:
            annotated_image = annotated_images.get("mobilenet") or next(iter(annotated_images.values()), "")

        if lite_mode:
            return {
                "mode": "image",
                "main_detection": main_detection,
                "annotated_image": annotated_image,
            }

        payload = {
            "mode": "image",
            "detections": detections,
            "fruit_counts": fruit_counts,
            "sampled_frames": 1,
            "main_detection": main_detection,
            "annotated_image": annotated_image,
            "annotated_images": annotated_images,
        }
        payload["debug"] = dbg
        return payload
    except Exception as exc:
        return JSONResponse({"error": f"Image prediction failed: {exc}"}, status_code=500)


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        return JSONResponse({"error": "Please upload a video."}, status_code=400)

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(tmp_path, "wb") as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    await file.close()

    cap = None
    writer = None

    try:
        load_assets()
        import cv2

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Cannot read video."}, status_code=400)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            return JSONResponse({"error": "Invalid video size."}, status_code=400)

        out_dir = STATIC_DIR / "tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"annotated_{uuid.uuid4().hex}.mp4"
        out_raw_name = f"annotated_{uuid.uuid4().hex}_raw.mp4"
        out_path = out_dir / out_name
        out_raw_path = out_dir / out_raw_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = float(fps) / float(VIDEO_FRAME_STEP)
        writer = cv2.VideoWriter(str(out_raw_path), fourcc, out_fps, (w, h))
        if not writer.isOpened():
            return JSONResponse({"error": "Cannot open output video writer."}, status_code=500)

        detections_all: List[dict] = []
        fruit_counts: Dict[str, dict] = {}
        dbg = []
        sampled_frames = 0
        first_annotated_img: Image.Image | None = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % VIDEO_FRAME_STEP != 0:
                frame_idx += 1
                continue
            sampled_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            det_list, _max_conf, dbg_det = detect_and_crop(img)
            if isinstance(dbg_det, dict):
                dbg.extend(dbg_det.get("attempts", []))

            frame_detections = []
            if det_list:
                filtered = [d for d in det_list if d["det_confidence"] >= DETECT_CONF] or det_list
                filtered = filtered[:VIDEO_MAX_BOXES]
                crop_count = len(filtered)
                crops = [d["crop"] for d in filtered]
                arr_batch = preprocess_batch(crops)
                models_batch = predict_models_batch(arr_batch, model_names=["mobilenet"])
                for det, models in zip(filtered, models_batch):
                    if not models:
                        continue
                    fruit = det["det_label"]
                    for m in models.values():
                        m["fruit"] = fruit
                        m["label"] = f"{fruit}_{m['quality']}"
                    det_item = {
                        "detection": {"box": det["box"], "label": fruit, "confidence": det["det_confidence"]},
                        "models": models,
                    }
                    frame_detections.append(det_item)
                    detections_all.append(det_item)

            annotated = annotate_pil(img, frame_detections)
            if first_annotated_img is None:
                first_annotated_img = annotated
            writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))
            frame_idx += 1

        if sampled_frames == 0:
            return JSONResponse({"error": "No frame read from video."}, status_code=400)

        # Ensure file handles are closed before encoding/serving.
        writer.release()
        writer = None
        cap.release()
        cap = None

        # Browser-friendly MP4 (H.264 + yuv420p + faststart)
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(out_raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                out_path = out_raw_path
                out_name = out_raw_name
            else:
                try:
                    os.remove(out_raw_path)
                except OSError:
                    pass
        else:
            out_path = out_raw_path
            out_name = out_raw_name

        main_detection = detections_all[0] if detections_all else None

        annotated_image = ""
        if first_annotated_img is not None:
            import base64

            buf = io.BytesIO()
            first_annotated_img.save(buf, format="PNG")
            annotated_image = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        return {
            "mode": "video",
            "detections": detections_all,
            "fruit_counts": fruit_counts,
            "sampled_frames": sampled_frames,
            "main_detection": main_detection,
            "annotated_image": annotated_image,
            "annotated_video": f"/static/tmp/{out_name}",
            "annotated_video_mime": "video/mp4",
            "debug": dbg,
        }
    except Exception as exc:
        return JSONResponse({"error": f"Video prediction failed: {exc}"}, status_code=500)
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass



@app.post("/start_video_stream")
async def start_video_stream(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        return JSONResponse({"error": "Please upload a video."}, status_code=400)

    job_id = uuid.uuid4().hex
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(tmp_path, "wb") as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    await file.close()
    _stream_jobs[job_id] = tmp_path
    return {"job_id": job_id, "stream_url": f"/stream_video/{job_id}"}


@app.post("/start_camera_stream")
async def start_camera_stream(source: int = 0):
    if source not in (-1, 0, 1, 2):
        return JSONResponse({"error": "Camera source must be -1, 0, 1, or 2."}, status_code=400)

    chosen = source
    debug = None
    if source == -1:
        chosen, debug = _pick_best_camera_source([0, 1, 2])
        if chosen is None:
            return JSONResponse({"error": "Cannot open usable camera source 0/1/2.", "debug": debug}, status_code=400)

    job_id = uuid.uuid4().hex
    _camera_jobs[job_id] = int(chosen)
    return {"job_id": job_id, "stream_url": f"/stream_camera/{job_id}", "source": int(chosen), "debug": debug}


@app.get("/camera_raw")
def camera_raw(source: int = -1):
    import cv2

    candidates = [source] if source >= 0 else [0, 1, 2]
    cap = None
    selected = None
    if source == -1:
        selected, _debug = _pick_best_camera_source([0, 1, 2])
        if selected is not None:
            cap = _open_cv_camera(int(selected))
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                selected = None
    else:
        for s in candidates:
            try:
                c = _open_cv_camera(int(s))
                if c.isOpened():
                    cap = c
                    selected = int(s)
                    break
                c.release()
            except Exception:
                continue

    if cap is None:
        return JSONResponse({"error": "Cannot open camera source 0/1/2."}, status_code=400)

    target_fps = max(1.0, float(CAMERA_STREAM_TARGET_FPS))
    frame_interval = 1.0 / target_fps
    jpeg_quality = max(40, min(95, int(CAMERA_STREAM_JPEG_QUALITY)))

    def _gen():
        next_due = time.perf_counter()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if CAMERA_STREAM_OUTPUT_MAX_WIDTH > 0 and frame.shape[1] > CAMERA_STREAM_OUTPUT_MAX_WIDTH:
                    scale = CAMERA_STREAM_OUTPUT_MAX_WIDTH / float(frame.shape[1])
                    out_w = CAMERA_STREAM_OUTPUT_MAX_WIDTH
                    out_h = max(2, int(round(frame.shape[0] * scale)))
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if ok:
                    payload = encoded.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n" + payload + b"\r\n"
                    )

                next_due += frame_interval
                sleep_s = next_due - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 0.05))
                else:
                    next_due = time.perf_counter()
        finally:
            try:
                cap.release()
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if selected is not None:
        headers["X-Camera-Source"] = str(selected)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@app.get("/stream_video/{job_id}")
def stream_video(job_id: str):
    tmp_path = _stream_jobs.get(job_id)
    if not tmp_path or not os.path.exists(tmp_path):
        return JSONResponse({"error": "Invalid or expired stream job."}, status_code=404)

    cfg = StreamConfig(
        video_frame_step=VIDEO_FRAME_STEP,
        stream_target_fps=STREAM_TARGET_FPS,
        stream_jpeg_quality=STREAM_JPEG_QUALITY,
        stream_detect_every=STREAM_DETECT_EVERY,
        stream_yolo_imgsz=STREAM_YOLO_IMGSZ,
        stream_min_conf=DETECT_CONF,
        stream_max_boxes=STREAM_MAX_BOXES,
        stream_classify_every=STREAM_CLASSIFY_EVERY,
        stream_classify_max_boxes=STREAM_CLASSIFY_MAX_BOXES,
        stream_output_max_width=STREAM_OUTPUT_MAX_WIDTH,
    )

    def _cleanup():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        _stream_jobs.pop(job_id, None)

    stream_iter = stream_video_frames(
        tmp_path=tmp_path,
        cfg=cfg,
        ensure_detector=ensure_detector,
        load_assets=load_assets,
        detect_and_crop=detect_and_crop,
        preprocess_batch=preprocess_batch,
        predict_models_batch=predict_models_batch,
        attach_quality_from_previous=attach_quality_from_previous,
        annotate_cv2=annotate_cv2,
        models_ref=_models,
        cleanup=_cleanup,
    )

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        stream_iter,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@app.get("/stream_camera/{job_id}")
def stream_camera(job_id: str):
    camera_source = _camera_jobs.get(job_id)
    if camera_source is None:
        return JSONResponse({"error": "Invalid or expired camera stream job."}, status_code=404)

    cfg = StreamConfig(
        video_frame_step=VIDEO_FRAME_STEP,
        stream_target_fps=CAMERA_STREAM_TARGET_FPS,
        stream_jpeg_quality=CAMERA_STREAM_JPEG_QUALITY,
        stream_detect_every=CAMERA_STREAM_DETECT_EVERY,
        stream_yolo_imgsz=CAMERA_STREAM_YOLO_IMGSZ,
        stream_min_conf=DETECT_CONF,
        stream_max_boxes=CAMERA_STREAM_MAX_BOXES,
        stream_classify_every=CAMERA_STREAM_CLASSIFY_EVERY,
        stream_classify_max_boxes=CAMERA_STREAM_CLASSIFY_MAX_BOXES,
        stream_output_max_width=CAMERA_STREAM_OUTPUT_MAX_WIDTH,
        capture_width=CAMERA_CAPTURE_WIDTH,
        capture_height=CAMERA_CAPTURE_HEIGHT,
        capture_fps=CAMERA_CAPTURE_FPS,
        capture_buffer_size=CAMERA_CAPTURE_BUFFER_SIZE,
        async_inference=CAMERA_ASYNC_INFERENCE,
    )

    def _cleanup():
        _camera_jobs.pop(job_id, None)

    stream_iter = stream_camera_frames(
        camera_source=camera_source,
        cfg=cfg,
        ensure_detector=ensure_detector,
        load_assets=load_assets,
        detect_and_crop=detect_and_crop,
        preprocess_batch=preprocess_batch,
        predict_models_batch=predict_models_batch,
        attach_quality_from_previous=attach_quality_from_previous,
        annotate_cv2=annotate_cv2,
        models_ref=_models,
        cleanup=_cleanup,
    )

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        stream_iter,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)








































