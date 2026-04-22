const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const inputHint = document.getElementById("inputHint");
const globalResult = document.getElementById("globalResult");

const inputImage = document.getElementById("inputImage");
const inputVideo = document.getElementById("inputVideo");
const outputImage = document.getElementById("outputImage");
const outputVideo = document.getElementById("outputVideo");
const outputStream = document.getElementById("outputStream");

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
const BROWSER_CAM_INTERVAL_MS = 650;

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

function showLiveVideo(el, stream) {
  if (!el) return;
  el.srcObject = stream;
  el.muted = true;
  el.playsInline = true;
  el.autoplay = true;
  el.classList.remove("hidden");
  el.play().catch(() => {});
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
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 \u1ea3nh. Ch\u1ecdn CNN ho\u1eb7c MobileNet tr\u01b0\u1edbc khi d\u1ef1 \u0111o\u00e1n.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. H\u00e3y ch\u1ecdn t\u1ec7p, ch\u1ecdn model r\u1ed3i b\u1ea5m D\u1ef1 \u0111o\u00e1n.", "empty");
  } else if (mode === "video") {
    inputCard?.classList.remove("hidden");
    fileInput.classList.remove("hidden");
    fileInput.accept = "video/*";
    modelSwitch?.classList.add("hidden");
    outResult?.classList.add("hidden");
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 video.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. H\u00e3y ch\u1ecdn t\u1ec7p v\u00e0 b\u1ea5m D\u1ef1 \u0111o\u00e1n.", "empty");
  } else {
    inputCard?.classList.add("hidden");
    fileInput.classList.add("hidden");
    modelSwitch?.classList.add("hidden");
    outResult?.classList.add("hidden");
    setText(inputHint, "\u0110ang \u1edf ch\u1ebf \u0111\u1ed9 camera realtime.");
    setGlobalResult("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3. B\u1ea5m D\u1ef1 \u0111o\u00e1n \u0111\u1ec3 ch\u1ea1y camera.", "empty");
  }
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
    throw new Error(`API ${res.status}: ${raw || "Ph?n h?i r?ng t? server"}`);
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

async function predictFrameBlob(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("selected_model", "mobilenet");
  const res = await fetch("/predict_image", { method: "POST", body: fd });
  return parseJsonResponse(res);
}

function isMobileBrowser() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent || "");
}

async function startBrowserCameraLoop() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Tr\u00ecnh duy\u1ec7t kh\u00f4ng h\u1ed7 tr\u1ee3 camera tr\u1ef1c ti\u1ebfp.");
  }

  stopBrowserCameraLoop();
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  browserCamStream = await navigator.mediaDevices.getUserMedia(constraints);
  showLiveVideo(inputVideo, browserCamStream);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Kh\u00f4ng kh\u1edfi t\u1ea1o \u0111\u01b0\u1ee3c canvas.");

  browserCamTimer = setInterval(async () => {
    if (!browserCamStream || browserCamBusy) return;

    const w = inputVideo.videoWidth || 0;
    const h = inputVideo.videoHeight || 0;
    if (!w || !h) return;

    browserCamBusy = true;
    try {
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(inputVideo, 0, 0, w, h);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
      if (!blob) throw new Error("Kh\u00f4ng \u0111\u1ecdc \u0111\u01b0\u1ee3c frame camera.");

      const data = await predictFrameBlob(blob);
      if (data.annotated_image) {
        showMedia(outputImage, data.annotated_image);
        hideMedia(outputVideo);
        hideMedia(outputStream);
      }

      setGlobalResult("\u0110ang ch\u1ea1y camera \u0111i\u1ec7n tho\u1ea1i tr\u1ef1c ti\u1ebfp...", "ok");
    } catch (err) {
      setGlobalResult(`L?i g?i API: ${err.message || err}`, "error");
    } finally {
      browserCamBusy = false;
    }
  }, BROWSER_CAM_INTERVAL_MS);
}

async function startAutoCameraStream() {
  if (isMobileBrowser()) {
    try {
      await startBrowserCameraLoop();
      return;
    } catch {
      // fallback sang ngu?n camera server-side
    }
  }

  let stream = null;
  try {
    stream = await startCameraStream(0);
  } catch {
    try {
      stream = await startCameraStream(1);
    } catch {
      stream = await startCameraStream(2);
    }
  }

  const streamUrl = stream.stream_url.startsWith("http")
    ? stream.stream_url
    : new URL(stream.stream_url, window.location.origin).toString();
  showMedia(outputStream, streamUrl);
  hideMedia(outputVideo);
  hideMedia(outputImage);
  setGlobalResult("\u0110ang stream camera \u0111\u00e3 qua m\u00f4 h\u00ecnh...", "ok");
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
        setGlobalResult("Camera tr?c ti?p tr?n ?i?n tho?i c?n HTTPS ho?c localhost.", "error");
      } else {
        setGlobalResult(`L?i g?i API: ${err.message || err}`, "error");
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
      setGlobalResult(`L?i g?i API: ${err.message || err}`, "error");
    }
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  showMedia(outputVideo, objectUrl, file.type || "video/mp4");
  hideMedia(outputImage);
  hideMedia(outputStream);
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
    setGlobalResult(`L?i g?i API: ${err.message || err}`, "error");
  }
});

syncInputByMode();
