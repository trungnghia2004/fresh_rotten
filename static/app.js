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
const modelRadios = document.querySelectorAll('input[name="modelPick"]');
const modelSwitch = document.getElementById("modelSwitch");
const inputCard = document.getElementById("inputCard");

const defaultCountsText = "t\u01b0\u01a1i 0 / h\u1ecfng 0 (crops 0)";
let latestMode = "image";
let latestModels = { cnn: null, mobilenet: null };
let latestAnnotatedImages = { cnn: null, mobilenet: null };

let browserCamStream = null;
let browserCamTimer = null;
let browserCamBusy = false;
const CAMERA_TARGET_FPS = 30;
const BROWSER_CAM_INTERVAL_MS = Math.max(20, Math.round(1000 / CAMERA_TARGET_FPS));
const REMOTE_SEND_ASPECT = 16 / 9;
const REMOTE_SEND_WIDTH = 1280;
const REMOTE_SEND_HEIGHT = 720;
const REMOTE_SEND_JPEG_QUALITY = 0.78;
const REMOTE_YOLO_IMGSZ = 640;
const REMOTE_HFLIP = true;
const IS_LOCAL_HOST = ["localhost", "127.0.0.1"].includes(window.location.hostname);

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

function drawOverlayDetections(detections = [], frameSize = null, mirror = false) {
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
  outputCamOverlay.style.transform = mirror ? "scaleX(-1)" : "none";
  outputCamOverlay.classList.remove("hidden");

  const ctx = outputCamOverlay.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, displayW, displayH);
  if (!Array.isArray(detections) || detections.length === 0) return;

  const fw = Math.max(1, Number(frameSize?.width || REMOTE_SEND_WIDTH));
  const fh = Math.max(1, Number(frameSize?.height || REMOTE_SEND_HEIGHT));
  const sx = displayW / fw;
  const sy = displayH / fh;

  ctx.lineWidth = 2;
  ctx.font = "16px Segoe UI";
  ctx.textBaseline = "top";

  for (const d of detections) {
    const box = d?.box || [];
    if (!Array.isArray(box) || box.length !== 4) continue;
    const x1 = Math.round(box[0] * sx);
    const y1 = Math.round(box[1] * sy);
    const x2 = Math.round(box[2] * sx);
    const y2 = Math.round(box[3] * sy);
    const w = Math.max(1, x2 - x1);
    const h = Math.max(1, y2 - y1);

    const fruit = String(d?.fruit || "unknown");
    const quality = String(d?.quality || "unknown");
    const label = `${fruit}_${quality}`;

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
  const checked = document.querySelector('input[name="modelPick"]:checked');
  return checked ? checked.value : null;
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
  latestModels = { cnn: null, mobilenet: null };
  latestAnnotatedImages = { cnn: null, mobilenet: null };
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
    modelSwitch?.classList.remove("hidden");
    outResult?.classList.remove("hidden");
    rawCamBtn?.classList.add("hidden");
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 \u1ea3nh. Ch\u1ecdn CNN ho\u1eb7c MobileNet tr\u01b0\u1edbc khi d\u1ef1 \u0111o\u00e1n.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. H\u00e3y ch\u1ecdn t\u1ec7p, ch\u1ecdn model r\u1ed3i b\u1ea5m D\u1ef1 \u0111o\u00e1n.", "empty");
  } else if (mode === "video") {
    inputCard?.classList.remove("hidden");
    fileInput.classList.remove("hidden");
    fileInput.accept = "video/*";
    modelSwitch?.classList.add("hidden");
    outResult?.classList.add("hidden");
    rawCamBtn?.classList.add("hidden");
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 video.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. H\u00e3y ch\u1ecdn t\u1ec7p v\u00e0 b\u1ea5m D\u1ef1 \u0111o\u00e1n.", "empty");
  } else {
    inputCard?.classList.add("hidden");
    fileInput.classList.add("hidden");
    modelSwitch?.classList.add("hidden");
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

async function predictFrameBlob(blob, imgsz) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("selected_model", "mobilenet");
  fd.append("allow_no_detection", "1");
  fd.append("lite", "1");
  fd.append("lite_render", "0");
  fd.append("imgsz", String(imgsz));
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

    browserCamBusy = true;
    try {
      canvas.width = w;
      canvas.height = h;

      if (isRemote && REMOTE_HFLIP) {
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(inputVideo, sx, sy, sw, sh, 0, 0, w, h);
        ctx.restore();
      } else {
        ctx.drawImage(inputVideo, sx, sy, sw, sh, 0, 0, w, h);
      }

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", isRemote ? REMOTE_SEND_JPEG_QUALITY : 0.85),
      );
      if (!blob) throw new Error("Kh\u00f4ng \u0111\u1ecdc \u0111\u01b0\u1ee3c frame camera.");

      const data = await predictFrameBlob(blob, REMOTE_YOLO_IMGSZ);
      if (isRemote) {
        drawOverlayDetections(data.detections || [], data.frame_size || null, REMOTE_HFLIP);
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

modelRadios.forEach((el) => el.addEventListener("change", updateCounts));
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
    if (!pickedModel) {
      setGlobalResult("Vui l\u00f2ng ch\u1ecdn CNN ho\u1eb7c MobileNet tr\u01b0\u1edbc khi d\u1ef1 \u0111o\u00e1n \u1ea3nh.", "error");
      return;
    }

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
      let mbData = data.models?.mobilenet;
      const det = data.main_detection || (Array.isArray(data.detections) && data.detections[0]);
      if (det) {
        cnnData = det.models?.cnn || cnnData;
        mbData = det.models?.mobilenet || mbData;
      }

      latestModels = { cnn: cnnData || null, mobilenet: mbData || null };
      latestAnnotatedImages = {
        cnn: data.annotated_images?.cnn || annotated || null,
        mobilenet: data.annotated_images?.mobilenet || annotated || null,
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
