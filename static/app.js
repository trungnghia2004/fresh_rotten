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
const cameraControls = document.getElementById("cameraControls");
const cameraSource = document.getElementById("cameraSource");

const defaultCountsText = "tươi 0 / hỏng 0 (crops 0)";
let latestModels = { cnn: null, mobilenet: null };
let latestMode = "image";
let browserCamStream = null;
let browserCamTimer = null;
let browserCamBusy = false;
const BROWSER_CAM_INTERVAL_MS = 650;

function setText(el, value) { if (el) el.textContent = value; }

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

function resetAll() {
  stopBrowserCameraLoop();
  hideMedia(inputImage); hideMedia(inputVideo);
  hideMedia(outputImage); hideMedia(outputVideo); hideMedia(outputStream);
  setText(outCounts, defaultCountsText);
  latestModels = { cnn: null, mobilenet: null };
}

function setGlobalResult(text, state = "ok") {
  if (!globalResult) return;
  globalResult.textContent = text;
  globalResult.className = `result ${state}`;
}

function syncInputByMode() {
  const mode = getMode();
  latestMode = mode;
  resetAll();
  if (mode !== "camera") fileInput.value = "";

  if (mode === "image") {
    fileInput.classList.remove("hidden");
    cameraControls?.classList.add("hidden");
    fileInput.accept = "image/*";
    setText(inputHint, "Đang ở chế độ ảnh.");
    modelSwitch?.classList.remove("hidden");
    outResult?.classList.remove("hidden");
  } else if (mode === "video") {
    fileInput.classList.remove("hidden");
    cameraControls?.classList.add("hidden");
    fileInput.accept = "video/*";
    setText(inputHint, "Đang ở chế độ video.");
    modelSwitch?.classList.add("hidden");
    outResult?.classList.add("hidden");
  } else {
    fileInput.classList.add("hidden");
    cameraControls?.classList.remove("hidden");
    setText(inputHint, "Đang ở chế độ camera realtime.");
    modelSwitch?.classList.add("hidden");
    outResult?.classList.add("hidden");
  }
  if (mode === "camera") {
    setGlobalResult("Chưa có kết quả. Hãy chọn nguồn camera và bấm Dự đoán.", "empty");
  } else {
    setGlobalResult("Chưa có kết quả. Hãy chọn tệp và bấm Dự đoán.", "empty");
  }
}

function updateCounts() {
  if (latestMode === "video") {
    setText(outCounts, "");
    return;
  }
  const picked = document.querySelector('input[name="modelPick"]:checked')?.value || "cnn";
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
  setText(outCounts, `tươi ${fresh ?? 0} / hỏng ${rotten ?? 0} (crops ${cropCount ?? 0})`);
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
  setGlobalResult("Đã tải tệp. Bấm Dự đoán để chạy.", "ok");
});

async function startVideoStream(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/start_video_stream", { method: "POST", body: fd });
  const raw = await res.text();
  let data = {};
  try { data = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`API ${res.status}: ${raw || "Phản hồi rỗng từ server"}`); }
  if (!res.ok) throw new Error(data?.error || data?.detail || `HTTP ${res.status}`);
  return data;
}

async function startCameraStream(source) {
  const src = Number.parseInt(source, 10);
  const res = await fetch(`/start_camera_stream?source=${Number.isFinite(src) ? src : 0}`, { method: "POST" });
  const raw = await res.text();
  let data = {};
  try { data = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`API ${res.status}: ${raw || "Phản hồi rỗng từ server"}`); }
  if (!res.ok) throw new Error(data?.error || data?.detail || `HTTP ${res.status}`);
  return data;
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

async function predictFrameBlob(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  const res = await fetch("/predict_image", { method: "POST", body: fd });
  const raw = await res.text();
  let data = {};
  try { data = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`API ${res.status}: ${raw || "Phản hồi rỗng từ server"}`); }
  if (!res.ok) throw new Error(data?.error || data?.detail || `HTTP ${res.status}`);
  return data;
}

async function startBrowserCameraLoop() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Trình duyệt không hỗ trợ camera trực tiếp.");
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
  hideMedia(inputImage);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Không khởi tạo được canvas.");

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
      if (!blob) throw new Error("Không đọc được frame camera.");
      const data = await predictFrameBlob(blob);

      if (data.annotated_image) {
        showMedia(outputImage, data.annotated_image);
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
      latestModels = { cnn: cnnData, mobilenet: mbData };
      updateCounts();
      setGlobalResult("Đang chạy camera điện thoại trực tiếp...", "ok");
    } catch (err) {
      setGlobalResult(`Lỗi gọi API: ${err.message || err}`, "error");
    } finally {
      browserCamBusy = false;
    }
  }, BROWSER_CAM_INTERVAL_MS);
}

async function postFile(url, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(url, { method: "POST", body: fd });
  const raw = await res.text();
  let data = {};
  try { data = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`API ${res.status}: ${raw || "Phản hồi rỗng từ server"}`); }
  if (!res.ok) throw new Error(data?.error || data?.detail || `HTTP ${res.status}`);
  return data;
}

predictBtn.addEventListener("click", async () => {
  const mode = getMode();
  latestMode = mode;
  const endpoint = mode === "image" ? "/predict_image" : "/predict_video";

  setGlobalResult("Đang xử lý...", "ok");

  if (mode === "camera") {
    hideMedia(outputImage);
    hideMedia(outputVideo);
    hideMedia(inputImage);
    hideMedia(outputStream);
    try {
      const source = cameraSource?.value ?? "0";
      if (source === "browser") {
        await startBrowserCameraLoop();
      } else {
        stopBrowserCameraLoop();
        hideMedia(inputVideo);
        const stream = await startCameraStream(source);
        const streamUrl = stream.stream_url.startsWith("http")
          ? stream.stream_url
          : new URL(stream.stream_url, window.location.origin).toString();
        showMedia(outputStream, streamUrl);
        setGlobalResult("Đang stream camera đã qua mô hình...", "ok");
      }
    } catch (err) {
      if (String(err).toLowerCase().includes("secure") || window.location.protocol !== "https:") {
        setGlobalResult("Camera trực tiếp trên điện thoại cần HTTPS hoặc localhost.", "error");
      } else {
        setGlobalResult(`Lỗi gọi API: ${err.message || err}`, "error");
      }
    }
    return;
  }

  const file = fileInput.files[0];
  if (!file) {
    setGlobalResult("Hãy chọn tệp trước khi dự đoán.", "error");
    return;
  }
  const objectUrl = URL.createObjectURL(file);

  if (mode === "video") {
    showMedia(outputVideo, objectUrl, file.type || "video/mp4");
    hideMedia(outputImage);
    hideMedia(outputStream);
    setGlobalResult("Đang tải video và khởi tạo stream...", "ok");

    try {
      const stream = await startVideoStream(file);
      const streamUrl = stream.stream_url.startsWith("http")
        ? stream.stream_url
        : new URL(stream.stream_url, window.location.origin).toString();
      showMedia(outputStream, streamUrl);
      hideMedia(outputVideo);
      hideMedia(outputImage);
      setGlobalResult("Đang stream video đã qua mô hình...", "ok");
    } catch (err) {
      setGlobalResult(`Lỗi gọi API: ${err.message || err}`, "error");
    }
    return;
  }

  try {
    const data = await postFile(endpoint, file);
    const annotated = data.annotated_image || null;
    const annotatedVideo = data.annotated_video || null;

    if (annotatedVideo) {
      const mime = data.annotated_video_mime || "video/mp4";
      const videoUrl = annotatedVideo.startsWith("http")
        ? annotatedVideo
        : new URL(annotatedVideo, window.location.origin).toString();
      showMedia(outputVideo, videoUrl, mime);
      hideMedia(outputImage);
      hideMedia(outputStream);
    } else if (annotated) {
      showMedia(outputImage, annotated);
      hideMedia(outputVideo);
      hideMedia(outputStream);
    } else if (mode === "image") {
      showMedia(outputImage, objectUrl);
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
    latestModels = { cnn: cnnData, mobilenet: mbData };

    updateCounts();
    setGlobalResult("Đã có kết quả.", "ok");
  } catch (err) {
    setGlobalResult(`Lỗi gọi API: ${err.message || err}`, "error");
  }
});

syncInputByMode();
