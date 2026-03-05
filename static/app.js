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

const defaultCountsText = "tươi 0 / hỏng 0 (crops 0)";

let latestModels = { cnn: null, mobilenet: null };
let latestMode = "image";

function setText(el, value) {
  if (el) el.textContent = value;
}

function hideMedia(el) {
  if (!el) return;
  if (el.tagName === "VIDEO") {
    el.pause();
    el.removeAttribute("src");
    el.load();
  } else {
    el.removeAttribute("src");
  }
  el.classList.add("hidden");
}

function showMedia(el, url, mime) {
  if (!el) return;
  if (el.tagName === "VIDEO" && mime) {
    el.type = mime;
  }
  el.src = url;
  if (el.tagName === "VIDEO") {
    el.muted = true;
    el.load();
    el.play().catch(() => {});
  }
  el.classList.remove("hidden");
}

function getMode() {
  const checked = document.querySelector('input[name="uploadMode"]:checked');
  return checked ? checked.value : "image";
}

function resetAll() {
  hideMedia(inputImage);
  hideMedia(inputVideo);
  hideMedia(outputImage);
  hideMedia(outputVideo);
  hideMedia(outputStream);
  setText(outCounts, defaultCountsText);
  latestModels = { cnn: null, mobilenet: null };
}

function setGlobalResult(text, state = "ok") {
  globalResult.textContent = text;
  globalResult.className = `result ${state}`;
}

function syncInputByMode() {
  const mode = getMode();
  latestMode = mode;
  fileInput.value = "";
  resetAll();

  if (mode === "image") {
    fileInput.accept = "image/*";
    inputHint.textContent = "Đang ở chế độ ảnh.";
    if (modelSwitch) modelSwitch.classList.remove("hidden");
    if (outResult) outResult.classList.remove("hidden");
  } else {
    fileInput.accept = "video/*";
    inputHint.textContent = "Đang ở chế độ video.";
    if (modelSwitch) modelSwitch.classList.add("hidden");
    if (outResult) outResult.classList.add("hidden");
  }

  setGlobalResult("Chưa có kết quả. Hãy chọn tệp và bấm Dự đoán.", "empty");
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

document.querySelectorAll('input[name="uploadMode"]').forEach((el) => {
  el.addEventListener("change", syncInputByMode);
});

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

async function postFile(url, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(url, { method: "POST", body: fd });
  const raw = await res.text();
  let data = {};
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`API ${res.status}: ${raw || "Phản hồi rỗng từ server"}`);
  }
  if (!res.ok) {
    throw new Error(data?.error || data?.detail || `HTTP ${res.status}`);
  }
  return data;
}

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setGlobalResult("Hãy chọn tệp trước khi dự đoán.", "error");
    return;
  }

  const mode = getMode();
  latestMode = mode;
  const endpoint = mode === "image" ? "/predict_image" : "/predict_video";
  const objectUrl = URL.createObjectURL(file);

  setGlobalResult("Đang xử lý...", "ok");

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
    } else {
      showMedia(outputVideo, objectUrl);
      hideMedia(outputImage);
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
