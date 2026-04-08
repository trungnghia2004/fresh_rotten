from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

# YOLO only: detect fruit boxes and optionally save crops
DETECT_MODEL = Path(r"F:\group23_22001611_22001624\weights\yolo_fruits_and_vegetables_v3.pt")


def main():
    p = argparse.ArgumentParser(description="Detect fruits with YOLO (no quality classification).")
    p.add_argument("image", type=Path, help="Input image")
    p.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    p.add_argument("--save-crops", action="store_true", help="Save cropped detections")
    args = p.parse_args()

    model = YOLO(str(DETECT_MODEL))
    img = Image.open(args.image).convert("RGB")
    res = model(np.array(img), imgsz=args.imgsz, verbose=False)[0]

    names = model.model.names  # type: ignore
    if not len(res.boxes):
        print("No detections.")
        return

    # chọn nhãn có conf cao nhất
    max_idx = int(np.argmax(res.boxes.conf.cpu().numpy()))
    top_cls = int(res.boxes.cls[max_idx])
    top_label = names.get(top_cls, str(top_cls))
    top_conf = float(res.boxes.conf[max_idx])
    print(f"Top label: {top_label} conf={top_conf:.3f}")

    # giữ tất cả box có cùng nhãn với nhãn cao nhất
    keep = []
    for i, (box, cls, score) in enumerate(zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf), 1):
        label = names.get(int(cls), str(int(cls)))
        if label != top_label:
            continue
        x1, y1, x2, y2 = map(int, box.cpu().numpy().tolist())
        conf = float(score)
        keep.append((i, label, conf, (x1, y1, x2, y2)))

    print(f"Kept {len(keep)} boxes for label '{top_label}'")
    for i, label, conf, (x1, y1, x2, y2) in keep:
        print(f"Box {i}: {label} conf={conf:.3f} at {(x1, y1, x2, y2)}")
        if args.save_crops:
            crop = img.crop((x1, y1, x2, y2))
            out_crop = args.image.parent / f"crop_{i}_{label}.png"
            crop.save(out_crop)


if __name__ == "__main__":
    main()

