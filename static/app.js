const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const inputHint = document.getElementById("inputHint");
const globalResult = document.getElementById("globalResult");

const panels = {
  cnn: {
    inputImage: document.getElementById("cnnInputImage"),
    inputVideo: document.getElementById("cnnInputVideo"),
    outputImage: document.getElementById("cnnOutputImage"),
    outputVideo: document.getElementById("cnnOutputVideo"),
    fruit: document.getElementById("cnnFruit"),
    quality: document.getElementById("cnnQuality"),
    confidence: document.getElementById("cnnConfidence"),
    sampled: document.getElementById("cnnSampled"),
    counts: document.getElementById("cnnCounts"),
  },
  mobilenet: {
    inputImage: document.getElementById("mobileInputImage"),
    inputVideo: document.getElementById("mobileInputVideo"),
    outputImage: document.getElementById("mobileOutputImage"),
    outputVideo: document.getElementById("mobileOutputVideo"),
    fruit: document.getElementById("mobileFruit"),
    quality: document.getElementById("mobileQuality"),
    confidence: document.getElementById("mobileConfidence"),
    sampled: document.getElementById("mobileSampled"),
    counts: document.getElementById("mobileCounts"),
  },
};

function getMode() {
  const checked = document.querySelector('input[name="uploadMode"]:checked');
  return checked ? checked.value : "image";
}

function setGlobalResult(text, state = "ok") {
  globalResult.textContent = text;
  globalResult.className = `result ${state}`;
}

function hideMedia(el) {
  el.classList.add("hidden");
  el.removeAttribute("src");
}

function resetPanel(panel) {
  hideMedia(panel.inputImage);
  hideMedia(panel.inputVideo);
  hideMedia(panel.outputImage);
  hideMedia(panel.outputVideo);
  panel.fruit.textContent = "-";
  panel.quality.textContent = "-";
  panel.confidence.textContent = "-";
  panel.sampled.textContent = "0";
  if (panel.counts) panel.counts.textContent = "tươi 0 / hỏng 0";
}

function resetAllPanels() {
  resetPanel(panels.cnn);
  resetPanel(panels.mobilenet);
}

function syncInputByMode() {
  const mode = getMode();
  fileInput.value = "";
  resetAllPanels();

  if (mode === "image") {
    fileInput.accept = "image/*";
    inputHint.textContent = "Đang ở chế độ ảnh.";
  } else {
    fileInput.accept = "video/*";
    inputHint.textContent = "Đang ở chế độ video.";
  }

  setGlobalResult("Chưa có kết quả. Hãy chọn tệp và bấm Dự đoán.", "empty");
}

function setPanelMedia(panel, mode, objectUrl, target) {
  const imageEl = target === "input" ? panel.inputImage : panel.outputImage;
  const videoEl = target === "input" ? panel.inputVideo : panel.outputVideo;

  if (mode === "image") {
    imageEl.src = objectUrl;
    imageEl.classList.remove("hidden");
    videoEl.classList.add("hidden");
  } else {
    videoEl.src = objectUrl;
    videoEl.classList.remove("hidden");
    imageEl.classList.add("hidden");
  }
}

function fillPanelResult(panel, modelData, sampledFrames, fruitOverride) {
  panel.fruit.textContent = fruitOverride || modelData?.fruit || "-";
  panel.quality.textContent = modelData?.quality || "-";
  const confidence = typeof modelData?.confidence === "number" ? modelData.confidence.toFixed(4) : "-";
  panel.confidence.textContent = confidence;
  const sampled = modelData?.sampled_frames ?? sampledFrames ?? 0;
  panel.sampled.textContent = String(sampled);

  const qc = modelData?.quality_counts || {};
  let fresh = qc.fresh;
  let rotten = qc.rotten;
  if (typeof fresh !== "number" || typeof rotten !== "number") {
    fresh = modelData?.quality === "fresh" ? 1 : 0;
    rotten = modelData?.quality === "rotten" ? 1 : 0;
  }
  if (panel.counts) panel.counts.textContent = `tươi ${fresh} / hỏng ${rotten}`;
}

async function postFile(url, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(url, { method: "POST", body: fd });
  const raw = await res.text();
  let data = null;

  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`API ${res.status}: ${raw || "Phản hồi rỗng từ server"}`);
  }

  if (!res.ok) {
    const msg = data?.error || data?.detail || `HTTP ${res.status}`;
    throw new Error(String(msg));
  }
  return data;
}

document.querySelectorAll('input[name="uploadMode"]').forEach((el) => {
  el.addEventListener("change", syncInputByMode);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  resetAllPanels();
  if (!file) return;

  const mode = getMode();
  const objectUrl = URL.createObjectURL(file);

  setPanelMedia(panels.cnn, mode, objectUrl, "input");
  setPanelMedia(panels.mobilenet, mode, objectUrl, "input");
  setGlobalResult("Đã tải tệp. Bấm Dự đoán để chạy cả CNN và MobileNet.", "ok");
});

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setGlobalResult("Hãy chọn tệp trước khi dự đoán.", "error");
    return;
  }

  const mode = getMode();
  const endpoint = mode === "image" ? "/predict_image" : "/predict_video";
  const objectUrl = URL.createObjectURL(file);

  setGlobalResult("Đang xử lý...", "ok");

  try {
    const data = await postFile(endpoint, file);

    let cnnData = data.models?.cnn;
    let mbData = data.models?.mobilenet;
    let fruitOverride = null;
    if (Array.isArray(data.detections) && data.detections.length > 0) {
      const det = data.detections[0];
      fruitOverride = det?.detection?.label || null;
      cnnData = det?.models?.cnn;
      mbData = det?.models?.mobilenet;
    }

    setPanelMedia(panels.cnn, mode, objectUrl, "output");
    setPanelMedia(panels.mobilenet, mode, objectUrl, "output");

    fillPanelResult(panels.cnn, cnnData, data.sampled_frames, fruitOverride);
    fillPanelResult(panels.mobilenet, mbData, data.sampled_frames, fruitOverride);

    setGlobalResult("Đã có kết quả để so sánh CNN và MobileNet.", "ok");
  } catch (err) {
    setGlobalResult(`Lỗi gọi API: ${err.message || err}`, "error");
  }
});

syncInputByMode();
