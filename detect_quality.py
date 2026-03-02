from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import tensorflow as tf

# Defaults
CWD = Path(__file__).parent
DEFAULT_DET = CWD / "full_model.pt"
DEFAULT_CNN = CWD / "cnn_best.keras"
DEFAULT_MB = CWD / "mobilenet_fruit_quality.keras"
DEFAULT_CLASSES = CWD / "class_names.json"
IMAGE_SIZE: Tuple[int, int] = (224, 224)


def load_class_names(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(x) for x in data]


def preprocess(img: Image.Image) -> np.ndarray:
    arr = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(arr, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def classify_quality(models: dict, arr: np.ndarray, class_names: List[str]) -> dict:
    results = {}
    for name, model in models.items():
        prob = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(prob))
        label = class_names[idx]
        fruit, quality = label.rsplit("_", 1)
        results[name] = {
            "label": label,
            "fruit": fruit,
            "quality": quality,
            "confidence": float(np.max(prob)),
        }
    return results


def draw_boxes(orig: Image.Image, det, per_box_results, out_path: Path):
    img = orig.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    colors = {
        "cnn": (0, 200, 255, 180),
        "mobilenet": (120, 255, 120, 180),
    }

    for (xyxy, cls_name, conf_det), res in zip(det, per_box_results):
        x1, y1, x2, y2 = map(int, xyxy)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 180, 0), width=3)
        y_text = y1 + 4
        header = f"{cls_name} ({conf_det:.2f})"
        draw.rectangle([x1, y1, x1 + 320, y1 + 22 + 22 * len(res)], fill=(0, 0, 0, 180))
        draw.text((x1 + 4, y_text), header, fill=(255, 255, 255), font=font)
        y_text += 22
        for model_name, info in res.items():
            color = colors.get(model_name, (200, 200, 255, 180))
            line = f"{model_name}: {info['quality']} ({info['confidence']:.3f})"
            draw.text((x1 + 4, y_text), line, fill=color, font=font)
            y_text += 22

    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Detect fruit boxes then classify quality.")
    ap.add_argument("image", type=Path, help="Path to input image")
    ap.add_argument("--det", type=Path, default=DEFAULT_DET, help="Detection model .pt (YOLO)")
    ap.add_argument("--cnn", type=Path, default=DEFAULT_CNN)
    ap.add_argument("--mobilenet", type=Path, default=DEFAULT_MB)
    ap.add_argument("--class-names", type=Path, default=DEFAULT_CLASSES)
    ap.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    ap.add_argument("--topn", type=int, default=5, help="Max boxes to process")
    ap.add_argument("--out", type=Path, default=Path("det_quality.png"))
    args = ap.parse_args()

    # Load models
    det_model = YOLO(str(args.det))
    class_names = load_class_names(args.class_names)

    models = {}
    if args.cnn.exists():
        models["cnn"] = tf.keras.models.load_model(args.cnn)
    if args.mobilenet.exists():
        models["mobilenet"] = tf.keras.models.load_model(args.mobilenet)
    if not models:
        raise RuntimeError("No quality models found.")

    # Detection
    det_results = det_model(args.image, verbose=False)[0]
    names = det_model.model.names  # type: ignore

    boxes_info = []
    for box, cls, conf in zip(det_results.boxes.xyxy, det_results.boxes.cls, det_results.boxes.conf):
        if float(conf) < args.conf:
            continue
        boxes_info.append((box.tolist(), names[int(cls)], float(conf)))
        if len(boxes_info) >= args.topn:
            break

    if not boxes_info:
        print("No detections above threshold.")
        return

    # Classify quality per box
    per_box_results = []
    with Image.open(args.image) as orig:
        for box in boxes_info:
            x1, y1, x2, y2 = map(int, box[0])
            crop = orig.crop((x1, y1, x2, y2))
            arr = preprocess(crop)
            per_box_results.append(classify_quality(models, arr, class_names))

        out_path = draw_boxes(orig, boxes_info, per_box_results, args.out)

    print("Detections + quality:")
    for (xyxy, cls_name, conf_det), res in zip(boxes_info, per_box_results):
        print(f"{cls_name} box {xyxy} conf {conf_det:.3f}")
        for mn, info in res.items():
            print(f"  {mn}: {info['quality']} ({info['confidence']:.3f})")

    print(f"Saved annotated image: {out_path}")


if __name__ == "__main__":
    main()
