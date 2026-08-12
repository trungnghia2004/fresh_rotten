const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const rawCamBtn = document.getElementById("rawCamBtn");
const inputHint = document.getElementById("inputHint");
const globalResult = document.getElementById("globalResult");

const inputImage = document.getElementById("inputImage");
const inputVideo = document.getElementById("inputVideo");
const outputImage = document.getElementById("outputImage");
const outputVideo = document.getElementById("outputVideo");
const outputStream = document.getElementById("outputStream");
const outputCamVideo = document.getElementById("outputCamVideo");
const outputCamOverlay = document.getElementById("outputCamOverlay");

const outCounts = document.getElementById("outCounts");
const outResult = document.getElementById("outResult");
const inputCard = document.getElementById("inputCard");

const defaultCountsText = "t\u01b0\u01a1i 0 / h\u1ecfng 0 (crops 0)";
let latestMode = "image";
let latestModels = { cnn: null };
let latestAnnotatedImages = { cnn: null };

let browserCamStream = null;
let browserCamTimer = null;
let browserCamBusy = false;
const CAMERA_TARGET_FPS = 30;
const BROWSER_CAM_INTERVAL_MS = Math.max(20, Math.round(1000 / CAMERA_TARGET_FPS));
const REMOTE_SEND_ASPECT = 16 / 9;
const REMOTE_SEND_WIDTH = 640;
const REMOTE_SEND_HEIGHT = 360;
const REMOTE_SEND_JPEG_QUALITY = 0.62;
const REMOTE_YOLO_IMGSZ = 384;
const REMOTE_HFLIP = false;
const REMOTE_LITE_MAX_BOXES = 12;
const REMOTE_LITE_CONF = 0.40;
const CAM_DET_HOLD_MS = 160;
const CAM_DET_SMOOTH_ALPHA = 0.82;
const CAM_DET_DEDUP_IOU = 0.65;
const CAM_DET_MIN_AREA = 700;
const CAM_TRACK_MAX_CENTER_DIST = 160;
const CAM_TRACK_MISS_MAX = 4;
const CAM_TRACK_VELOCITY_DECAY = 0.82;
const CAM_TRACK_MIN_SCORE = 0.16;
const CAM_DETECT_INTERVAL_MS = 90;
const CAM_CLASSIFY_EVERY = 3;

let camLastDetections = [];
let camLastFrameSize = null;
let camLastCaptureMeta = null;
let camLastDetectionTs = 0;
let camLastApiRequestTs = 0;
let camDetectSeq = 0;
let camTracks = [];
let camNextTrackId = 1;
let camRenderRaf = 0;
const IS_LOCAL_HOST = ["localhost", "127.0.0.1"].includes(window.location.hostname);

function boxIou(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return 0;
  const ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
  const bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function boxCenter(box) {
  return [
    (Number(box?.[0] || 0) + Number(box?.[2] || 0)) / 2,
    (Number(box?.[1] || 0) + Number(box?.[3] || 0)) / 2,
  ];
}

function centerDistance(a, b) {
  const [ax, ay] = boxCenter(a);
  const [bx, by] = boxCenter(b);
  return Math.hypot(ax - bx, ay - by);
}

function shiftBox(box, dx, dy) {
  if (!Array.isArray(box) || box.length !== 4) return [0, 0, 0, 0];
  return [
    Math.round(Number(box[0]) + dx),
    Math.round(Number(box[1]) + dy),
    Math.round(Number(box[2]) + dx),
    Math.round(Number(box[3]) + dy),
  ];
}

function smoothBox(prevBox, nextBox, alpha) {
  return [
    Math.round(prevBox[0] * (1 - alpha) + nextBox[0] * alpha),
    Math.round(prevBox[1] * (1 - alpha) + nextBox[1] * alpha),
    Math.round(prevBox[2] * (1 - alpha) + nextBox[2] * alpha),
    Math.round(prevBox[3] * (1 - alpha) + nextBox[3] * alpha),
  ];
}

function predictTrackBox(track, nowTs = performance.now()) {
  const lastUpdateTs = Number(track?.lastUpdateTs || nowTs);
  const elapsed = Math.max(0, Math.min(180, nowTs - lastUpdateTs));
  const vx = Number(track?.vx || 0);
  const vy = Number(track?.vy || 0);
  return shiftBox(track?.box || [0, 0, 0, 0], vx * elapsed, vy * elapsed);
}

function trackingScore(cur, track, nowTs = performance.now()) {
  const curBox = cur?.box;
  const prevBox = predictTrackBox(track, nowTs);
  if (!Array.isArray(curBox) || !Array.isArray(prevBox) || curBox.length !== 4 || prevBox.length !== 4) return -Infinity;

  const iou = boxIou(curBox, prevBox);
  const dist = centerDistance(curBox, prevBox);
  const distNorm = Math.max(0, 1 - dist / CAM_TRACK_MAX_CENTER_DIST);
  const sameFruit = cur?.fruit && track?.fruit && cur.fruit === track.fruit;
  const sameQuality = cur?.quality && track?.quality && cur.quality === track.quality;

  return iou * 0.65 + distNorm * 0.3 + (sameFruit ? 0.08 : 0) + (sameQuality ? 0.03 : 0);
}

function toCameraTrack(det, nowTs = performance.now()) {
  return {
    id: camNextTrackId++,
    fruit: det?.fruit || "unknown",
    quality: det?.quality || "unknown",
    confidence: Number(det?.confidence || 0),
    box: det?.box || [0, 0, 0, 0],
    vx: 0,
    vy: 0,
    missCount: 0,
    lastUpdateTs: nowTs,
  };
}

function trackToDetection(track, nowTs = performance.now()) {
  return {
    box: predictTrackBox(track, nowTs),
    fruit: track.fruit,
    quality: track.quality,
    confidence: track.confidence,
    trackId: track.id,
    missCount: track.missCount,
  };
}

function getRenderableCameraDetections(nowTs = performance.now()) {
  return camTracks
    .map((track) => trackToDetection(track, nowTs))
    .sort((a, b) => Number(a?.box?.[0] || 0) - Number(b?.box?.[0] || 0));
}

function updateCameraTracks(currentDetections, previousTracks, nowTs = performance.now()) {
  const current = Array.isArray(currentDetections) ? currentDetections : [];
  const previous = Array.isArray(previousTracks) ? previousTracks : [];
  if (current.length === 0) {
    return previous
      .map((track) => {
        const missCount = Number(track?.missCount || 0) + 1;
        if (missCount > CAM_TRACK_MISS_MAX) return null;
        const nextBox = predictTrackBox(track, nowTs);
        return {
          ...track,
          box: nextBox,
          vx: Number(track?.vx || 0) * CAM_TRACK_VELOCITY_DECAY,
          vy: Number(track?.vy || 0) * CAM_TRACK_VELOCITY_DECAY,
          missCount,
          lastUpdateTs: nowTs,
          confidence: Math.max(0, Number(track?.confidence || 0) * 0.94),
        };
      })
      .filter(Boolean)
      .sort((a, b) => Number(a?.box?.[0] || 0) - Number(b?.box?.[0] || 0));
  }

  if (previous.length === 0) {
    return [...current]
      .sort((a, b) => Number(a?.box?.[0] || 0) - Number(b?.box?.[0] || 0))
      .map((det) => toCameraTrack(det, nowTs));
  }

  const usedTracks = new Set();
  const sortedCur = [...current].sort((a, b) => Number(a?.box?.[0] || 0) - Number(b?.box?.[0] || 0));
  const out = [];

  for (const cur of sortedCur) {
    const curBox = cur?.box;
    if (!Array.isArray(curBox) || curBox.length !== 4) {
      out.push(toCameraTrack(cur, nowTs));
      continue;
    }

    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < previous.length; i++) {
      if (usedTracks.has(i)) continue;
      const track = previous[i];
      const score = trackingScore(cur, track, nowTs);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (bestIdx < 0 || bestScore < CAM_TRACK_MIN_SCORE) {
      out.push(toCameraTrack(cur, nowTs));
      continue;
    }

    usedTracks.add(bestIdx);
    const best = previous[bestIdx];
    const predictedBox = predictTrackBox(best, nowTs);
    const dist = centerDistance(cur.box, predictedBox);
    const alpha = dist > 90 ? 0.94 : CAM_DET_SMOOTH_ALPHA;
    const smoothedBox = smoothBox(predictedBox, cur.box, alpha);
    const quality = cur?.quality && cur.quality !== "unknown" ? cur.quality : (best?.quality || "unknown");
    const fruit = cur?.fruit || best?.fruit || "unknown";
    const [px, py] = boxCenter(best.box);
    const [nx, ny] = boxCenter(smoothedBox);
    const dt = Math.max(16, nowTs - Number(best?.lastUpdateTs || (nowTs - 16)));
    const vx = (nx - px) / dt;
    const vy = (ny - py) / dt;
    out.push({
      ...best,
      fruit,
      quality,
      confidence: Number(cur?.confidence || best?.confidence || 0),
      box: smoothedBox,
      vx: Number(best?.vx || 0) * 0.25 + vx * 0.75,
      vy: Number(best?.vy || 0) * 0.25 + vy * 0.75,
      missCount: 0,
      lastUpdateTs: nowTs,
    });
  }

  for (let i = 0; i < previous.length; i++) {
    if (usedTracks.has(i)) continue;
    const prev = previous[i];
    const missCount = Number(prev?.missCount || 0) + 1;
    if (missCount > CAM_TRACK_MISS_MAX) continue;
    out.push({
      ...prev,
      box: predictTrackBox(prev, nowTs),
      vx: Number(prev?.vx || 0) * CAM_TRACK_VELOCITY_DECAY,
      vy: Number(prev?.vy || 0) * CAM_TRACK_VELOCITY_DECAY,
      missCount,
      lastUpdateTs: nowTs,
      confidence: Math.max(0, Number(prev?.confidence || 0) * 0.94),
    });
  }

  return out.sort((a, b) => Number(a?.box?.[0] || 0) - Number(b?.box?.[0] || 0));
}

function dedupeDetections(detections, iouThreshold = CAM_DET_DEDUP_IOU) {
  if (!Array.isArray(detections) || detections.length <= 1) return detections || [];
  const sorted = [...detections].sort((a, b) => Number(b?.confidence || 0) - Number(a?.confidence || 0));
  const kept = [];
  for (const d of sorted) {
    const box = d?.box;
    if (!Array.isArray(box) || box.length !== 4) continue;
    const area = Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
    if (area < CAM_DET_MIN_AREA) continue;
    let isDup = false;
    for (const k of kept) {
      if (boxIou(box, k.box) >= iouThreshold) {
        isDup = true;
        break;
      }
    }
    if (!isDup) kept.push(d);
  }
  return kept;
}

function setText(el, value) {
  if (el) el.textContent = value;
}

function hideMedia(el) {
  if (!el) return;
  if (el.tagName === "VIDEO") {
    el.pause();
    if (el.srcObject) {
      const tracks = el.srcObject.getTracks?.() || [];
      tracks.forEach((t) => t.stop());
      el.srcObject = null;
    }
    el.removeAttribute("src");
    el.load();
  } else {
    el.removeAttribute("src");
  }
  el.classList.add("hidden");
}

function showMedia(el, url, mime) {
  if (!el) return;
  if (el.tagName === "VIDEO" && mime) el.type = mime;
  el.src = url;
  if (el.tagName === "VIDEO") {
    el.muted = true;
    el.playsInline = true;
    el.load();
    el.play().catch(() => {});
  }
  el.classList.remove("hidden");
}

function showLiveVideo(el, stream, mirror = false) {
  if (!el) return;
  el.srcObject = stream;
  el.muted = true;
  el.playsInline = true;
  el.autoplay = true;
  el.style.transform = mirror ? "scaleX(-1)" : "none";
  el.classList.remove("hidden");
  el.play().catch(() => {});
}

function clearOverlay() {
  if (!outputCamOverlay) return;
  const ctx = outputCamOverlay.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, outputCamOverlay.width, outputCamOverlay.height);
}

function hideOverlay() {
  clearOverlay();
  outputCamOverlay?.classList.add("hidden");
}

function renderCameraOverlayFrame(nowTs = performance.now()) {
  if (!browserCamStream || getMode() !== "camera" || !outputCamVideo || outputCamVideo.classList.contains("hidden")) {
    camRenderRaf = 0;
    return;
  }

  const detections = getRenderableCameraDetections(nowTs);
  camLastDetections = detections;
  drawOverlayDetections(
    detections,
    camLastFrameSize || null,
    REMOTE_HFLIP,
    camLastCaptureMeta || null,
  );
  camRenderRaf = requestAnimationFrame(renderCameraOverlayFrame);
}

function ensureCameraOverlayLoop() {
  if (!camRenderRaf) {
    camRenderRaf = requestAnimationFrame(renderCameraOverlayFrame);
  }
}

function drawOverlayDetections(detections = [], frameSize = null, mirror = false, captureMeta = null) {
  if (!outputCamOverlay || !outputCamVideo) return;
  const displayW = outputCamVideo.clientWidth || 0;
  const displayH = outputCamVideo.clientHeight || 0;
  if (!displayW || !displayH) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = Math.max(2, Math.round(displayW * dpr));
  const ch = Math.max(2, Math.round(displayH * dpr));
  if (outputCamOverlay.width !== cw || outputCamOverlay.height !== ch) {
    outputCamOverlay.width = cw;
    outputCamOverlay.height = ch;
  }
  // Do not mirror canvas text. We mirror box coordinates manually below.
  outputCamOverlay.style.transform = "none";
  outputCamOverlay.classList.remove("hidden");

  const ctx = outputCamOverlay.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, displayW, displayH);
  if (!Array.isArray(detections) || detections.length === 0) return;

  // Mapping chain:
  // detection box coords (on sent frame fw/fh)
  // -> source camera coords (srcW/srcH with crop sx0,sy0,sw0,sh0)
  // -> displayed video coords (object-fit contain)
  const fw = Math.max(1, Number(frameSize?.width || captureMeta?.fw || REMOTE_SEND_WIDTH));
  const fh = Math.max(1, Number(frameSize?.height || captureMeta?.fh || REMOTE_SEND_HEIGHT));
  const srcW = Math.max(1, Number(captureMeta?.srcW || outputCamVideo.videoWidth || fw));
  const srcH = Math.max(1, Number(captureMeta?.srcH || outputCamVideo.videoHeight || fh));
  const sx0 = Math.max(0, Number(captureMeta?.sx || 0));
  const sy0 = Math.max(0, Number(captureMeta?.sy || 0));
  const sw0 = Math.max(1, Number(captureMeta?.sw || srcW));
  const sh0 = Math.max(1, Number(captureMeta?.sh || srcH));

  const containScale = Math.min(displayW / srcW, displayH / srcH);
  const drawW = srcW * containScale;
  const drawH = srcH * containScale;
  const offX = (displayW - drawW) / 2;
  const offY = (displayH - drawH) / 2;

  ctx.lineWidth = 2;
  ctx.font = "16px Segoe UI";
  ctx.textBaseline = "top";

  for (const d of detections) {
    const box = d?.box || [];
    if (!Array.isArray(box) || box.length !== 4) continue;
    const bx1 = Number(box[0]);
    const by1 = Number(box[1]);
    const bx2 = Number(box[2]);
    const by2 = Number(box[3]);

    // back to source coords from cropped-sent frame coords
    const srcX1 = sx0 + (bx1 / fw) * sw0;
    const srcY1 = sy0 + (by1 / fh) * sh0;
    const srcX2 = sx0 + (bx2 / fw) * sw0;
    const srcY2 = sy0 + (by2 / fh) * sh0;

    let x1 = Math.round(offX + srcX1 * containScale);
    const y1 = Math.round(offY + srcY1 * containScale);
    let x2 = Math.round(offX + srcX2 * containScale);
    const y2 = Math.round(offY + srcY2 * containScale);
    if (mirror) {
      const nx1 = displayW - x2;
      const nx2 = displayW - x1;
      x1 = nx1;
      x2 = nx2;
    }
    const w = Math.max(1, x2 - x1);
    const h = Math.max(1, y2 - y1);

    const quality = String(d?.quality || "unknown");
    const label = (quality === "fresh" || quality === "rotten") ? quality : "unknown";

    ctx.strokeStyle = "#00e05a";
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.strokeRect(x1, y1, w, h);
    const tw = Math.ceil(ctx.measureText(label).width);
    const th = 20;
    const ty = Math.max(0, y1 - th - 2);
    ctx.fillRect(x1, ty, tw + 10, th);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + 5, ty + 2);
  }
}

function getMode() {
  const checked = document.querySelector('input[name="uploadMode"]:checked');
  return checked ? checked.value : "image";
}

function getPickedModel() {
  return "cnn";
}

function setGlobalResult(text, state = "ok") {
  if (!globalResult) return;
  globalResult.textContent = text;
  globalResult.className = `result ${state}`;
}

function stopBrowserCameraLoop() {
  if (browserCamTimer) {
    clearInterval(browserCamTimer);
    browserCamTimer = null;
  }
  if (camRenderRaf) {
    cancelAnimationFrame(camRenderRaf);
    camRenderRaf = 0;
  }
  if (browserCamStream) {
    const tracks = browserCamStream.getTracks?.() || [];
    tracks.forEach((t) => t.stop());
    browserCamStream = null;
  }
  browserCamBusy = false;
}

function resetAll() {
  stopBrowserCameraLoop();
  hideMedia(inputImage);
  hideMedia(inputVideo);
  hideMedia(outputImage);
  hideMedia(outputVideo);
  hideMedia(outputStream);
  hideMedia(outputCamVideo);
  hideOverlay();
  setText(outCounts, defaultCountsText);
  latestModels = { cnn: null };
  latestAnnotatedImages = { cnn: null };
  camLastDetections = [];
  camTracks = [];
  camNextTrackId = 1;
  camLastFrameSize = null;
  camLastCaptureMeta = null;
  camLastDetectionTs = 0;
  camLastApiRequestTs = 0;
  camDetectSeq = 0;
}

function syncInputByMode() {
  const mode = getMode();
  latestMode = mode;
  resetAll();

  if (mode !== "camera") fileInput.value = "";

  if (mode === "image") {
    inputCard?.classList.remove("hidden");
    fileInput.classList.remove("hidden");
    fileInput.accept = "image/*";
    outResult?.classList.remove("hidden");
    rawCamBtn?.classList.add("hidden");
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 \u1ea3nh. H\u1ec7 th\u1ed1ng ch\u1ec9 d\u00f9ng CNN.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. H\u00e3y ch\u1ecdn t\u1ec7p r\u1ed3i b\u1ea5m D\u1ef1 \u0111o\u00e1n.", "empty");
  } else if (mode === "video") {
    inputCard?.classList.remove("hidden");
    fileInput.classList.remove("hidden");
    fileInput.accept = "video/*";
    outResult?.classList.add("hidden");
    rawCamBtn?.classList.add("hidden");
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 video.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. H\u00e3y ch\u1ecdn t\u1ec7p v\u00e0 b\u1ea5m D\u1ef1 \u0111o\u00e1n.", "empty");
  } else {
    inputCard?.classList.add("hidden");
    fileInput.classList.add("hidden");
    outResult?.classList.add("hidden");
    if (IS_LOCAL_HOST) {
      rawCamBtn?.classList.remove("hidden");
    } else {
      rawCamBtn?.classList.add("hidden");
    }
    setText(inputHint, "\u0110ang ch\u1ea1y camera.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. B\u1ea5m D\u1ef1 \u0111o\u00e1n \u0111\u1ec3 ch\u1ea1y camera.", "empty");
  }
}

function startRawCameraPreview() {
  resetAll();
  const rawUrl = `/camera_raw?source=-1&_=${Date.now()}`;
  showMedia(outputStream, rawUrl);
  hideMedia(outputVideo);
  hideMedia(outputImage);
  hideMedia(outputCamVideo);
  hideOverlay();
  setGlobalResult("\u0110ang test camera g\u1ed1c (kh\u00f4ng qua m\u00f4 h\u00ecnh)...", "ok");
}

function updateCounts() {
  if (latestMode !== "image") {
    setText(outCounts, "");
    return;
  }

  const picked = getPickedModel();
  if (!picked) {
    setText(outCounts, defaultCountsText);
    return;
  }

  const modelData = latestModels[picked];
  if (!modelData) {
    setText(outCounts, defaultCountsText);
    return;
  }

  const qc = modelData.quality_counts || {};
  let fresh = qc.fresh;
  let rotten = qc.rotten;
  if (typeof fresh !== "number" || typeof rotten !== "number") {
    fresh = modelData.quality === "fresh" ? 1 : 0;
    rotten = modelData.quality === "rotten" ? 1 : 0;
  }
  const cropCount = modelData.crop_count ?? modelData.detections_count ?? 0;
  setText(outCounts, `t\u01b0\u01a1i ${fresh ?? 0} / h\u1ecfng ${rotten ?? 0} (crops ${cropCount ?? 0})`);

  const annotated = latestAnnotatedImages[picked];
  if (annotated) {
    showMedia(outputImage, annotated);
    hideMedia(outputVideo);
    hideMedia(outputStream);
  }
}

async function parseJsonResponse(res) {
  const raw = await res.text();
  let data = {};
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`API ${res.status}: ${raw || "Ph\u1ea3n h\u1ed3i r\u1ed7ng t\u1eeb server"}`);
  }
  if (!res.ok) throw new Error(data?.error || data?.detail || `HTTP ${res.status}`);
  return data;
}

async function startVideoStream(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/start_video_stream", { method: "POST", body: fd });
  return parseJsonResponse(res);
}

async function startCameraStream(source) {
  const res = await fetch(`/start_camera_stream?source=${source}`, { method: "POST" });
  return parseJsonResponse(res);
}

async function postFile(url, file, selectedModel = null) {
  const fd = new FormData();
  fd.append("file", file);
  if (selectedModel) fd.append("selected_model", selectedModel);
  const res = await fetch(url, { method: "POST", body: fd });
  return parseJsonResponse(res);
}

async function predictFrameBlob(blob, imgsz, classify = true) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("selected_model", "cnn");
  fd.append("allow_no_detection", "1");
  fd.append("lite", "1");
  fd.append("lite_render", "0");
  fd.append("lite_max_boxes", String(REMOTE_LITE_MAX_BOXES));
  fd.append("lite_conf", String(REMOTE_LITE_CONF));
  fd.append("imgsz", String(imgsz));
  fd.append("classify", classify ? "1" : "0");
  const res = await fetch("/predict_image", { method: "POST", body: fd });
  return parseJsonResponse(res);
}

async function startBrowserCameraLoop() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Tr\u00ecnh duy\u1ec7t kh\u00f4ng h\u1ed7 tr\u1ee3 camera tr\u1ef1c ti\u1ebfp.");
  }

  stopBrowserCameraLoop();
  const isRemote = !IS_LOCAL_HOST;
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: isRemote ? 1280 : 1280, min: 960 },
      height: { ideal: isRemote ? 720 : 720, min: 540 },
      frameRate: { ideal: CAMERA_TARGET_FPS, min: 24, max: 30 },
    },
  };

  browserCamStream = await navigator.mediaDevices.getUserMedia(constraints);
  showLiveVideo(inputVideo, browserCamStream, isRemote && REMOTE_HFLIP);
  if (isRemote) {
    showLiveVideo(outputCamVideo, browserCamStream, REMOTE_HFLIP);
    ensureCameraOverlayLoop();
    hideMedia(outputImage);
    hideMedia(outputVideo);
    hideMedia(outputStream);
  } else {
    hideMedia(outputCamVideo);
    hideOverlay();
  }

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Kh\u00f4ng kh\u1edfi t\u1ea1o \u0111\u01b0\u1ee3c canvas.");

  browserCamTimer = setInterval(async () => {
    if (!browserCamStream || browserCamBusy) return;

    const srcW = inputVideo.videoWidth || 0;
    const srcH = inputVideo.videoHeight || 0;
    if (!srcW || !srcH) return;

    let w = srcW;
    let h = srcH;
    let sx = 0;
    let sy = 0;
    let sw = srcW;
    let sh = srcH;

    if (isRemote) {
      w = REMOTE_SEND_WIDTH;
      h = REMOTE_SEND_HEIGHT;
      const srcAspect = srcW / srcH;
      if (srcAspect > REMOTE_SEND_ASPECT) {
        sh = srcH;
        sw = Math.max(2, Math.round(sh * REMOTE_SEND_ASPECT));
        sx = Math.max(0, Math.round((srcW - sw) / 2));
        sy = 0;
      } else {
        sw = srcW;
        sh = Math.max(2, Math.round(sw / REMOTE_SEND_ASPECT));
        sx = 0;
        sy = Math.max(0, Math.round((srcH - sh) / 2));
      }
    }

    const captureMeta = {
      srcW,
      srcH,
      sx,
      sy,
      sw,
      sh,
      fw: w,
      fh: h,
    };

    if (isRemote) {
      camLastCaptureMeta = captureMeta;
      camLastFrameSize = camLastFrameSize || { width: w, height: h };
      const loopNow = performance.now();
      if (loopNow - camLastApiRequestTs < CAM_DETECT_INTERVAL_MS) {
        return;
      }
      camLastApiRequestTs = loopNow;
    }

    browserCamBusy = true;
    try {
      canvas.width = w;
      canvas.height = h;

      // Keep inference frame unmirrored for stable geometry.
      // Mirroring is applied only on displayed video/overlay.
      ctx.drawImage(inputVideo, sx, sy, sw, sh, 0, 0, w, h);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", isRemote ? REMOTE_SEND_JPEG_QUALITY : 0.85),
      );
      if (!blob) throw new Error("Kh?ng ??c ???c frame camera.");

      const shouldClassify = !isRemote || camTracks.length === 0 || (camDetectSeq % CAM_CLASSIFY_EVERY === 0);
      camDetectSeq += 1;
      const data = await predictFrameBlob(blob, REMOTE_YOLO_IMGSZ, shouldClassify);
      if (isRemote) {
        const now = performance.now();
        const incomingRaw = Array.isArray(data.detections) ? data.detections : [];
        const incoming = dedupeDetections(incomingRaw);
        if (incoming.length > 0) {
          camTracks = updateCameraTracks(incoming, camTracks, now);
          camLastDetections = getRenderableCameraDetections(now);
          camLastFrameSize = data.frame_size || camLastFrameSize || { width: w, height: h };
          camLastCaptureMeta = captureMeta;
          camLastDetectionTs = now;
        } else {
          camTracks = updateCameraTracks([], camTracks, now);
          camLastDetections = getRenderableCameraDetections(now);
          if (camTracks.length === 0 && now - camLastDetectionTs > CAM_DET_HOLD_MS) {
            camLastDetections = [];
          }
          camLastFrameSize = data.frame_size || camLastFrameSize;
          camLastCaptureMeta = captureMeta;
        }

      } else if (data.annotated_image) {
        showMedia(outputImage, data.annotated_image);
        hideMedia(outputVideo);
        hideMedia(outputStream);
        hideMedia(outputCamVideo);
        hideOverlay();
      }

      setGlobalResult("\u0110ang ch\u1ea1y camera.", "ok");
    } catch (err) {
      setGlobalResult(`L\u1ed7i g\u1ecdi API: ${err.message || err}`, "error");
    } finally {
      browserCamBusy = false;
    }
  }, BROWSER_CAM_INTERVAL_MS);
}

async function startAutoCameraStream() {
  // On server machine, prefer server-side camera stream for higher FPS and steady updates.
  if (IS_LOCAL_HOST) {
    let stream = null;
    try {
      stream = await startCameraStream(-1);
    } catch {
      try {
        stream = await startCameraStream(0);
      } catch {
        try {
          stream = await startCameraStream(1);
        } catch {
          stream = await startCameraStream(2);
        }
      }
    }

    const streamUrl = stream.stream_url.startsWith("http")
      ? stream.stream_url
      : new URL(stream.stream_url, window.location.origin).toString();
    showMedia(outputStream, streamUrl);
    hideMedia(outputVideo);
    hideMedia(outputImage);
    hideMedia(outputCamVideo);
    hideOverlay();
    setGlobalResult("\\u0110ang stream camera \\u0111\\u00e3 qua m\\u00f4 h\\u00ecnh...", "ok");
    return;
  }

  if (window.isSecureContext && navigator.mediaDevices?.getUserMedia) {
    try {
      await startBrowserCameraLoop();
      return;
    } catch {
      // fallback sang nguồn camera server-side
    }
  }

  // On remote clients (ngrok/LAN), camera mode should use client camera only.
  throw new Error("Kh\u00f4ng truy c\u1eadp \u0111\u01b0\u1ee3c camera tr\u00ean m\u00e1y n\u00e0y. H\u00e3y c\u1ea5p quy\u1ec1n camera cho tr\u00ecnh duy\u1ec7t.");
}

document.querySelectorAll('input[name="uploadMode"]').forEach((el) => el.addEventListener("change", syncInputByMode));

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  resetAll();
  if (!file) return;

  const mode = getMode();
  const objectUrl = URL.createObjectURL(file);
  if (mode === "image") {
    showMedia(inputImage, objectUrl);
    hideMedia(inputVideo);
  } else {
    showMedia(inputVideo, objectUrl);
    hideMedia(inputImage);
  }
  setGlobalResult("\u0110\u00e3 t\u1ea3i t\u1ec7p. B\u1ea5m D\u1ef1 \u0111o\u00e1n \u0111\u1ec3 ch\u1ea1y.", "ok");
});

predictBtn.addEventListener("click", async () => {
  const mode = getMode();
  latestMode = mode;

  setGlobalResult("\u0110ang x\u1eed l\u00fd...", "ok");

  if (mode === "camera") {
    resetAll();
    try {
      await startAutoCameraStream();
    } catch (err) {
      if (String(err).toLowerCase().includes("secure") || window.location.protocol !== "https:") {
        setGlobalResult("Camera tr\u1ef1c ti\u1ebfp tr\u00ean \u0111i\u1ec7n tho\u1ea1i c\u1ea7n HTTPS ho\u1eb7c localhost.", "error");
      } else {
        setGlobalResult(`L\u1ed7i g\u1ecdi API: ${err.message || err}`, "error");
      }
    }
    return;
  }

  const file = fileInput.files[0];
  if (!file) {
    setGlobalResult("H\u00e3y ch\u1ecdn t\u1ec7p tr\u01b0\u1edbc khi d\u1ef1 \u0111o\u00e1n.", "error");
    return;
  }

  if (mode === "image") {
    const pickedModel = getPickedModel();
    try {
      const data = await postFile("/predict_image", file, pickedModel);
      const annotated = data.annotated_image || null;

      if (annotated) {
        showMedia(outputImage, annotated);
        hideMedia(outputVideo);
        hideMedia(outputStream);
        hideMedia(outputCamVideo);
        hideOverlay();
      }

      let cnnData = data.models?.cnn;
      const det = data.main_detection || (Array.isArray(data.detections) && data.detections[0]);
      if (det) {
        cnnData = det.models?.cnn || cnnData;
      }

      latestModels = { cnn: cnnData || null };
      latestAnnotatedImages = {
        cnn: data.annotated_images?.cnn || annotated || null,
      };

      updateCounts();
      setGlobalResult("\u0110\u00e3 c\u00f3 k\u1ebft qu\u1ea3.", "ok");
    } catch (err) {
      setGlobalResult(`L\u1ed7i g\u1ecdi API: ${err.message || err}`, "error");
    }
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  showMedia(outputVideo, objectUrl, file.type || "video/mp4");
  hideMedia(outputImage);
  hideMedia(outputStream);
  hideMedia(outputCamVideo);
  hideOverlay();
  setGlobalResult("\u0110ang t\u1ea3i video v\u00e0 kh\u1edfi t\u1ea1o stream...", "ok");

  try {
    const stream = await startVideoStream(file);
    const streamUrl = stream.stream_url.startsWith("http")
      ? stream.stream_url
      : new URL(stream.stream_url, window.location.origin).toString();
    showMedia(outputStream, streamUrl);
    hideMedia(outputVideo);
    hideMedia(outputImage);
    setGlobalResult("\u0110ang stream video \u0111\u00e3 qua m\u00f4 h\u00ecnh...", "ok");
  } catch (err) {
    setGlobalResult(`L\u1ed7i g\u1ecdi API: ${err.message || err}`, "error");
  }
});

rawCamBtn?.addEventListener("click", () => {
  if (getMode() !== "camera") return;
  startRawCameraPreview();
});

syncInputByMode();

