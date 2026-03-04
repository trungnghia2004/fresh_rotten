const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const inputHint = document.getElementById("inputHint");
const globalResult = document.getElementById("globalResult");

const inputImage = document.getElementById("inputImage");
const inputVideo = document.getElementById("inputVideo");
const outputImage = document.getElementById("outputImage");
const outputVideo = document.getElementById("outputVideo");
const outCounts = document.getElementById("outCounts");
const modelRadios = document.querySelectorAll('input[name="modelPick"]');

const defaultCountsText = "tươi 0 / hỏng 0 (crops 0)";

let latestModels = { cnn: null, mobilenet: null };
let latestMode = "image";
let latestAnnotated = null;

function setText(el, value) {
  if (el) el.textContent = value;
}

function hideMedia(el) {
  el.classList.add("hidden");
  el.removeAttribute("src");
}

function showMedia(el, url) {
  el.src = url;
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
  setText(outCounts, defaultCountsText);
  latestModels = { cnn: null, mobilenet: null };
  latestAnnotated = null;
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
  } else {
    fileInput.accept = "video/*";
    inputHint.textContent = "Đang ở chế độ video.";
  }
  setGlobalResult("Chưa có kết quả. Hãy chọn tệp và bấm Dự đoán.", "empty");
}

function updateCounts() {
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

    // chọn annotated image
    const annotated = data.annotated_image || objectUrl;
    latestAnnotated = annotated;
    if (mode === "image") {
      showMedia(outputImage, annotated);
      hideMedia(outputVideo);
    } else {
      showMedia(outputVideo, annotated);
      hideMedia(outputImage);
    }

    // lấy model data từ main_detection nếu có, else từ data.models
    let cnnData = data.models?.cnn;
    let mbData = data.models?.mobilenet;
    const det = data.main_detection || (Array.isArray(data.detections) && data.detections[0]);
    if (det) {
      cnnData = det.models?.cnn || cnnData;
      mbData = det.models?.mobilenet || mbData;
    }
    latestModels = { cnn: cnnData, mobilenet: mbData };
    updateCounts();
    setGlobalResult("Đã có kết quả. Chọn CNN/MobileNet để xem thống kê.", "ok");
  } catch (err) {
    setGlobalResult(`Lỗi gọi API: ${err.message || err}`, "error");
  }
});

syncInputByMode();

