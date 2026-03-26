from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --- CONFIG (chỉnh tại đây, không cần truyền tham số) ---
IMAGE_PATH = Path(r"F:\fresh_rotten\test\test_image.png")  # đổi sang ảnh bạn muốn test
MODEL_PATH = Path(r"F:\fresh_rotten\yolo_fruits_and_vegetables_v3.pt")
CONF_THRESH = 0.3
OUT_PATH = Path("yolo_boxes.png")
# -------------------------------------------------------

def draw_boxes(img_path: Path, model_path: Path, out_path: Path, conf: float):
    model = YOLO(str(model_path))
    res = model(img_path, conf=conf, verbose=False)[0]

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    names = model.model.names  # type: ignore
    for box, cls, score in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_name = names.get(int(cls), str(int(cls)))
        label = f"{cls_name} {float(score):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 180, 0, 255), width=3)
        # textbbox works across Pillow versions
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=(0, 0, 0, 200))
        draw.text((x1 + 3, y1 - th - 2), label, fill=(255, 255, 255), font=font)

    img.save(out_path)
    return out_path


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"IMAGE_PATH not found: {IMAGE_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    out = draw_boxes(IMAGE_PATH, MODEL_PATH, OUT_PATH, CONF_THRESH)
    print(f"Saved boxed image to {out}")


if __name__ == "__main__":
    main()
