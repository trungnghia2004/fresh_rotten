from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE: Tuple[int, int] = (224, 224)
ALLOWED_QUALITIES = {"fresh", "rotten"}


@dataclass
class ModelBundle:
    name: str
    model: tf.keras.Model
    class_names: List[str]
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]


def build_class_names(train_dir: str | Path) -> List[str]:
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_path}")

    labels: List[str] = []
    for fruit_dir in train_path.iterdir():
        if not fruit_dir.is_dir():
            continue
        fruit = fruit_dir.name.strip()
        for quality_dir in fruit_dir.iterdir():
            if not quality_dir.is_dir():
                continue
            quality = quality_dir.name.strip().lower()
            if quality in ALLOWED_QUALITIES:
                labels.append(f"{fruit}_{quality}")

    class_names = sorted(set(labels))
    if not class_names:
        raise ValueError(
            f"No valid class labels found in {train_path}. "
            "Expected structure train/<fruit>/<fresh|rotten>/..."
        )
    return class_names


def load_class_names_file(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return sorted(set(str(x).strip() for x in data if str(x).strip()))


def build_label_mappings(class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def load_model_bundle(model_path: str | Path, class_names: List[str], name: str) -> ModelBundle:
    model = tf.keras.models.load_model(model_path)
    label_to_idx, idx_to_label = build_label_mappings(class_names)
    return ModelBundle(
        name=name,
        model=model,
        class_names=class_names,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
    )


def preprocess_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def _activation_name(model: tf.keras.Model) -> str | None:
    last_layer = model.layers[-1] if model.layers else None
    activation = getattr(last_layer, "activation", None)
    if activation is None:
        return None
    return getattr(activation, "__name__", None)


def _to_probabilities(raw: np.ndarray, activation: str | None) -> np.ndarray:
    if activation == "softmax":
        probs = raw
    else:
        # Safe fallback for logits or numerically unstable outputs.
        shifted = raw - np.max(raw)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)
    probs = np.asarray(probs, dtype=np.float32)
    return probs / np.sum(probs)


def decode_prediction(bundle: ModelBundle, pred: np.ndarray) -> Tuple[str, str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    output_dim = pred.shape[0]
    class_count = len(bundle.class_names)
    activation = _activation_name(bundle.model)

    if output_dim == class_count:
        probabilities = _to_probabilities(pred, activation)
        class_idx = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
    elif output_dim == 1 and class_count == 2 and activation == "sigmoid":
        prob_pos = float(np.clip(pred[0], 0.0, 1.0))
        class_idx = 1 if prob_pos >= 0.5 else 0
        confidence = prob_pos if class_idx == 1 else 1.0 - prob_pos
    else:
        raise ValueError(
            f"Incompatible output dimension: model outputs {output_dim}, "
            f"but class_names has {class_count} classes."
        )

    label = bundle.idx_to_label[class_idx]
    if "_" not in label:
        raise ValueError(f"Invalid label format '{label}'. Expected '<fruit>_<quality>'.")
    fruit, quality = label.rsplit("_", 1)
    return fruit, quality, confidence


def predict_one(bundle: ModelBundle, image_array: np.ndarray) -> Tuple[str, str, float]:
    pred = bundle.model.predict(image_array, verbose=0)
    return decode_prediction(bundle, pred)


def print_prediction(model_name: str, fruit: str, quality: str, confidence: float) -> None:
    print(f"{model_name}:")
    print(f"    Quả: {fruit}")
    print(f"    Chất lượng: {quality}")
    print(f"    Độ tin cậy: {confidence:.4f}")


def existing_models(cnn_path: str | Path) -> List[Tuple[str, Path]]:
    path = Path(cnn_path)
    return [("CNN", path)] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Fruit quality inference with consistent label decoding.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--train-dir", default="train", help="Training directory for class order.")
    parser.add_argument("--class-names-file", default="class_names.json", help="JSON list of class names.")
    parser.add_argument("--cnn-model", default="cnn_best.keras", help="Path to CNN model.")
    args = parser.parse_args()

    class_names = load_class_names_file(args.class_names_file)
    if not class_names:
        class_names = build_class_names(args.train_dir)
    label_to_idx, idx_to_label = build_label_mappings(class_names)

    print("Class order (sorted):")
    print(class_names)
    print("label_to_idx:")
    print(label_to_idx)
    print("idx_to_label:")
    print(idx_to_label)
    print()

    image_array = preprocess_image(args.image)
    model_entries = existing_models(args.cnn_model)
    if not model_entries:
        raise FileNotFoundError("No model found. Check --cnn-model path.")

    for model_name, model_path in model_entries:
        bundle = load_model_bundle(model_path, class_names, model_name)
        fruit, quality, confidence = predict_one(bundle, image_array)
        print_prediction(model_name, fruit, quality, confidence)
        print()


if __name__ == "__main__":
    main()
