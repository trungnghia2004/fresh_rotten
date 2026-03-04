const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const inputHint = document.getElementById("inputHint");
const globalResult = document.getElementById("globalResult");

function setText(el, value) {
  if (el) el.textContent = value;
}

const panels = {
  cnn: {
    inputImage: document.getElementById("cnnInputImage"),
    inputVideo: document.getElementById("cnnInputVideo"),
    outputImage: document.getElementById("cnnOutputImage"),
    outputVideo: document.getElementById("cnnOutputVideo"),
    fruit: document.getElementById("cnnFruit"),
    sampled: document.getElementById("cnnSampled"),
    counts: document.getElementById("cnnCounts"),
  },
  mobilenet: {
    inputImage: document.getElementById("mobileInputImage"),
    inputVideo: document.getElementById("mobileInputVideo"),
    outputImage: document.getElementById("mobileOutputImage"),
    outputVideo: document.getElementById("mobileOutputVideo"),
    fruit: document.getElementById("mobileFruit"),
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
  setText(panel.fruit, "-");
  setText(panel.sampled, "0");
  setText(panel.counts, "tươi 0 / hỏng 0 (crops 0)");
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
    inputHint.textContent = "Ã„Âang Ã¡Â»Å¸ chÃ¡ÂºÂ¿ Ã„â€˜Ã¡Â»â„¢ Ã¡ÂºÂ£nh.";
  } else {
    fileInput.accept = "video/*";
    inputHint.textContent = "Ã„Âang Ã¡Â»Å¸ chÃ¡ÂºÂ¿ Ã„â€˜Ã¡Â»â„¢ video.";
  }

  setGlobalResult("ChÃ†Â°a cÃƒÂ³ kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£. HÃƒÂ£y chÃ¡Â»Ân tÃ¡Â»â€¡p vÃƒÂ  bÃ¡ÂºÂ¥m DÃ¡Â»Â± Ã„â€˜oÃƒÂ¡n.", "empty");
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
  setText(panel.fruit, fruitOverride || modelData?.fruit || "-");
  const sampled = modelData?.sampled_frames ?? sampledFrames ?? 0;
  setText(panel.sampled, String(sampled));

  const qc = modelData?.quality_counts || {};
  let fresh = qc.fresh;
  let rotten = qc.rotten;
  if (typeof fresh !== "number" || typeof rotten !== "number") {
    fresh = modelData?.quality === "fresh" ? 1 : 0;
    rotten = modelData?.quality === "rotten" ? 1 : 0;
  }
  const cropCount = modelData?.crop_count ?? modelData?.detections_count ?? 0;
  setText(panel.counts, `tươi ${fresh} / hỏng ${rotten} (crops ${cropCount})`);
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
    throw new Error(`API ${res.status}: ${raw || "PhÃ¡ÂºÂ£n hÃ¡Â»â€œi rÃ¡Â»â€”ng tÃ¡Â»Â« server"}`);
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
  setGlobalResult("Ã„ÂÃƒÂ£ tÃ¡ÂºÂ£i tÃ¡Â»â€¡p. BÃ¡ÂºÂ¥m DÃ¡Â»Â± Ã„â€˜oÃƒÂ¡n Ã„â€˜Ã¡Â»Æ’ chÃ¡ÂºÂ¡y cÃ¡ÂºÂ£ CNN vÃƒÂ  MobileNet.", "ok");
});

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setGlobalResult("HÃƒÂ£y chÃ¡Â»Ân tÃ¡Â»â€¡p trÃ†Â°Ã¡Â»â€ºc khi dÃ¡Â»Â± Ã„â€˜oÃƒÂ¡n.", "error");
    return;
  }

  const mode = getMode();
  const endpoint = mode === "image" ? "/predict_image" : "/predict_video";
  const objectUrl = URL.createObjectURL(file);

  setGlobalResult("Ã„Âang xÃ¡Â»Â­ lÃƒÂ½...", "ok");

  try {
    const data = await postFile(endpoint, file);

    let cnnData = data.models?.cnn;
    let mbData = data.models?.mobilenet;
    let fruitOverride = null;
    const det = data.main_detection || (Array.isArray(data.detections) && data.detections.length > 0 ? data.detections[0] : null);
    if (det) {
      fruitOverride = det?.detection?.label || null;
      cnnData = det?.models?.cnn;
      mbData = det?.models?.mobilenet;
    }

    setPanelMedia(panels.cnn, mode, objectUrl, "output");
    setPanelMedia(panels.mobilenet, mode, objectUrl, "output");

    fillPanelResult(panels.cnn, cnnData, data.sampled_frames, fruitOverride);
    fillPanelResult(panels.mobilenet, mbData, data.sampled_frames, fruitOverride);

    setGlobalResult("Ã„ÂÃƒÂ£ cÃƒÂ³ kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ Ã„â€˜Ã¡Â»Æ’ so sÃƒÂ¡nh CNN vÃƒÂ  MobileNet.", "ok");
  } catch (err) {
    setGlobalResult(`LÃ¡Â»â€”i gÃ¡Â»Âi API: ${err.message || err}`, "error");
  }
});

syncInputByMode();



