from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Union

import numpy as np
from PIL import Image


@dataclass
class StreamConfig:
    video_frame_step: int
    stream_target_fps: float
    stream_jpeg_quality: int
    stream_detect_every: int
    stream_yolo_imgsz: int
    stream_min_conf: float
    stream_max_boxes: int
    stream_classify_every: int
    stream_classify_max_boxes: int
    stream_output_max_width: int
    capture_width: int = 0
    capture_height: int = 0
    capture_fps: float = 0.0
    capture_buffer_size: int = 0



def _iou(a: List[int], b: List[int]) -> float:
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


def stabilize_labels_with_previous(filtered: List[dict], previous: List[dict]) -> List[dict]:
    if not filtered or not previous:
        return filtered

    out = []
    for cur in filtered:
        cur_box = cur.get("box")
        cur_label = cur.get("det_label")
        cur_conf = float(cur.get("det_confidence", 0.0))

        best_prev = None
        best_iou = 0.0
        for prev in previous:
            prev_det = prev.get("detection", {})
            prev_box = prev_det.get("box")
            if not prev_box:
                continue
            iou = _iou(cur_box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_prev = prev_det

        # Nếu box trùng mạnh với frame trước mà label mới yếu hơn, giữ label cũ.
        if best_prev is not None and best_iou >= 0.6:
            prev_label = best_prev.get("label")
            prev_conf = float(best_prev.get("confidence", 0.0))
            if prev_label and prev_label != cur_label and cur_conf <= (prev_conf + 0.12):
                cur = dict(cur)
                cur["det_label"] = prev_label

        out.append(cur)

    return out

def _open_capture(capture_source: Union[str, int], cfg: StreamConfig):
    import cv2

    if isinstance(capture_source, int):
        cap = cv2.VideoCapture(capture_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(capture_source)
        if cap.isOpened():
            if cfg.capture_buffer_size > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.capture_buffer_size)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            if cfg.capture_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture_width)
            if cfg.capture_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
            if cfg.capture_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, cfg.capture_fps)
    else:
        cap = cv2.VideoCapture(capture_source)
    return cap


def _stream_capture_frames(
    *,
    capture_source: Union[str, int],
    cfg: StreamConfig,
    ensure_detector: Callable[[], bool],
    load_assets: Callable[[], None],
    detect_and_crop: Callable[..., tuple],
    preprocess_batch: Callable[[List[Image.Image]], np.ndarray],
    predict_models_batch: Callable[..., List[Dict[str, dict]]],
    attach_quality_from_previous: Callable[[List[dict], List[dict]], None],
    annotate_cv2: Callable[[np.ndarray, List[dict]], np.ndarray],
    models_ref: Dict[str, object],
    cleanup: Callable[[], None],
) -> Iterator[bytes]:
    import cv2

    cap = None
    try:
        cap = _open_capture(capture_source, cfg)
        if not cap.isOpened():
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = cfg.capture_fps if cfg.capture_fps > 0 else 25.0

        dynamic_step = max(1, cfg.video_frame_step)
        if cfg.stream_target_fps > 0 and source_fps > cfg.stream_target_fps:
            dynamic_step = max(dynamic_step, int(math.ceil(source_fps / cfg.stream_target_fps)))

        output_fps = max(1.0, source_fps / float(dynamic_step))
        frame_interval = 1.0 / output_fps
        next_due = time.perf_counter()

        jpeg_quality = max(50, min(95, cfg.stream_jpeg_quality))
        frame_idx = 0
        last_frame_detections: List[dict] = []
        last_detect_frame_idx = -10**9
        last_classify_frame_idx = -10**9

        detect_enabled = ensure_detector()
        classify_enabled = False
        if detect_enabled:
            try:
                load_assets()
                classify_enabled = "mobilenet" in models_ref
            except Exception:
                classify_enabled = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % dynamic_step != 0:
                frame_idx += 1
                continue

            frame_detections = last_frame_detections

            if detect_enabled:
                try:
                    need_recompute = (not last_frame_detections) or ((frame_idx - last_detect_frame_idx) >= cfg.stream_detect_every)
                    if need_recompute:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        frame_detections = []
                        det_list, _max_conf, _dbg_det = detect_and_crop(img, imgsz=cfg.stream_yolo_imgsz)

                        if det_list:
                            filtered = [d for d in det_list if d["det_confidence"] >= cfg.stream_min_conf] or det_list
                            filtered = filtered[: cfg.stream_max_boxes]
                            filtered = stabilize_labels_with_previous(filtered, last_frame_detections)

                            if last_frame_detections:
                                strict_new_conf = min(0.95, cfg.stream_min_conf + 0.2)
                                stable = []
                                for d in filtered:
                                    has_match = False
                                    for p in last_frame_detections:
                                        pbox = p.get("detection", {}).get("box")
                                        if pbox and _iou(d["box"], pbox) >= 0.35:
                                            has_match = True
                                            break
                                    if has_match or float(d.get("det_confidence", 0.0)) >= strict_new_conf:
                                        stable.append(d)
                                if stable:
                                    filtered = stable

                            for det in filtered:
                                frame_detections.append(
                                    {
                                        "detection": {
                                            "box": det["box"],
                                            "label": det["det_label"],
                                            "confidence": det["det_confidence"],
                                        },
                                        "models": {},
                                    }
                                )

                            need_classify = classify_enabled and (
                                (not last_frame_detections)
                                or ((frame_idx - last_classify_frame_idx) >= cfg.stream_classify_every)
                            )
                            if need_classify:
                                classify_items = filtered[: cfg.stream_classify_max_boxes]
                                crops = [d["crop"] for d in classify_items]
                                arr_batch = preprocess_batch(crops)
                                models_batch = predict_models_batch(arr_batch, model_names=["mobilenet"])
                                for i, (det, models) in enumerate(zip(classify_items, models_batch)):
                                    if not models:
                                        continue
                                    fruit = det["det_label"]
                                    for m in models.values():
                                        m["fruit"] = fruit
                                        m["label"] = f"{fruit}_{m['quality']}"
                                    frame_detections[i]["models"] = models
                                last_classify_frame_idx = frame_idx

                            attach_quality_from_previous(frame_detections, last_frame_detections)

                        last_frame_detections = frame_detections
                        last_detect_frame_idx = frame_idx
                except Exception:
                    frame_detections = last_frame_detections

            render_frame = annotate_cv2(frame, frame_detections) if detect_enabled else frame
            if cfg.stream_output_max_width > 0 and render_frame.shape[1] > cfg.stream_output_max_width:
                scale = cfg.stream_output_max_width / float(render_frame.shape[1])
                out_w = cfg.stream_output_max_width
                out_h = max(2, int(round(render_frame.shape[0] * scale)))
                render_frame = cv2.resize(render_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            ok, encoded = cv2.imencode(".jpg", render_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                payload = encoded.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + payload + b"\r\n"
                )

            frame_idx += 1

            next_due += frame_interval
            sleep_s = next_due - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.05))
            else:
                next_due = time.perf_counter()
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cleanup()


def stream_video_frames(
    *,
    tmp_path: str,
    cfg: StreamConfig,
    ensure_detector: Callable[[], bool],
    load_assets: Callable[[], None],
    detect_and_crop: Callable[..., tuple],
    preprocess_batch: Callable[[List[Image.Image]], np.ndarray],
    predict_models_batch: Callable[..., List[Dict[str, dict]]],
    attach_quality_from_previous: Callable[[List[dict], List[dict]], None],
    annotate_cv2: Callable[[np.ndarray, List[dict]], np.ndarray],
    models_ref: Dict[str, object],
    cleanup: Callable[[], None],
) -> Iterator[bytes]:
    return _stream_capture_frames(
        capture_source=tmp_path,
        cfg=cfg,
        ensure_detector=ensure_detector,
        load_assets=load_assets,
        detect_and_crop=detect_and_crop,
        preprocess_batch=preprocess_batch,
        predict_models_batch=predict_models_batch,
        attach_quality_from_previous=attach_quality_from_previous,
        annotate_cv2=annotate_cv2,
        models_ref=models_ref,
        cleanup=cleanup,
    )


def stream_camera_frames(
    *,
    camera_source: int,
    cfg: StreamConfig,
    ensure_detector: Callable[[], bool],
    load_assets: Callable[[], None],
    detect_and_crop: Callable[..., tuple],
    preprocess_batch: Callable[[List[Image.Image]], np.ndarray],
    predict_models_batch: Callable[..., List[Dict[str, dict]]],
    attach_quality_from_previous: Callable[[List[dict], List[dict]], None],
    annotate_cv2: Callable[[np.ndarray, List[dict]], np.ndarray],
    models_ref: Dict[str, object],
    cleanup: Callable[[], None],
) -> Iterator[bytes]:
    return _stream_capture_frames(
        capture_source=camera_source,
        cfg=cfg,
        ensure_detector=ensure_detector,
        load_assets=load_assets,
        detect_and_crop=detect_and_crop,
        preprocess_batch=preprocess_batch,
        predict_models_batch=predict_models_batch,
        attach_quality_from_previous=attach_quality_from_previous,
        annotate_cv2=annotate_cv2,
        models_ref=models_ref,
        cleanup=cleanup,
    )


