import { AnimeGanRunner, type Backend, type SegmentationModel } from "./animegan";
import { MODELS, type AnimeModel } from "./models";
import "./styles.css";

type Status = "idle" | "loading" | "ready" | "error";

const icon = (name: "code" | "film" | "play" | "restart" | "sparkles" | "upload" | "x", size: number) => {
  const paths = {
    code: '<path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/>',
    film: '<rect width="20" height="20" x="2" y="2" rx="2.18" ry="2.18"/><line x1="7" x2="7" y1="2" y2="22"/><line x1="17" x2="17" y1="2" y2="22"/><line x1="2" x2="22" y1="12" y2="12"/><line x1="2" x2="7" y1="7" y2="7"/><line x1="2" x2="7" y1="17" y2="17"/><line x1="17" x2="22" y1="17" y2="17"/><line x1="17" x2="22" y1="7" y2="7"/>',
    play: '<polygon points="6 3 20 12 6 21 6 3"/>',
    restart: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    sparkles: '<path d="m12 3-1.9 4.8L5 10l5.1 2.2L12 17l1.9-4.8L19 10l-5.1-2.2Z"/><path d="m5 3-.6 1.4L3 5l1.4.6L5 7l.6-1.4L7 5l-1.4-.6Z"/><path d="m19 17-.9 2.1L16 20l2.1.9L19 23l.9-2.1L22 20l-2.1-.9Z"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  };
  return `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths[name]}</svg>`;
};

const root = document.querySelector<HTMLDivElement>("#root")!;
root.innerHTML = `
  <div class="app-shell">
    <header>
      <a class="brand" href="#" aria-label="Frame by Frame home"><span class="brand-mark">${icon("film", 19)}</span><span>FRAME <i>BY</i> FRAME</span></a>
      <div class="header-meta"><span class="privacy"><span></span> Runs entirely on your device</span><a href="https://github.com/ldenoue/animeganv3-browser" target="_blank" rel="noreferrer">${icon("code", 18)} Source</a></div>
    </header>
    <main>
      <section class="intro">
        <div class="eyebrow"><span>01</span> VIDEO TO ANIME</div>
        <h1>See your world<br><em>drawn differently.</em></h1>
        <p>Drop in a video and your selected model redraws each frame locally. Optionally enable MediaPipe to place detected people on chroma green first.</p>
      </section>
      <div class="drop-zone">
        <input id="file-input" type="file" accept="video/*">
        <span class="upload-icon">${icon("upload", 26)}</span>
        <strong>Drop your video here</strong>
        <span>or choose another input · MP4, WebM, MOV</span>
        <div class="source-actions">
          <label class="choose-button" for="file-input">Choose a video</label>
          <button class="example-button" type="button">Try example</button>
          <button class="webcam-button" type="button">Use webcam</button>
        </div>
        <span class="source-message" role="status"></span>
      </div>
      <section class="workspace" hidden>
        <div class="workspace-head">
          <div><span class="step">02</span><strong id="file-name"></strong></div>
          <button class="remove" type="button" aria-label="Remove video">${icon("x", 17)} Replace</button>
        </div>
        <div class="tiles">
          <article class="tile source-tile">
            <div class="tile-label"><span>ORIGINAL</span><small>INPUT</small></div>
            <video muted loop playsinline preload="auto" controls></video>
            <div class="crop-guide"><span>FULL HEIGHT · 256²</span></div>
          </article>
          <article class="tile result-tile">
            <div class="tile-label"><span id="result-model"></span><small id="result-backend">FULL FRAME</small></div>
            <canvas></canvas>
            <div class="result-empty">${icon("sparkles", 30)}<span>Your stylized frames<br>will appear here</span></div>
          </article>
        </div>
        <div class="control-bar">
          <div class="status status-idle"><span></span><span id="status-text">Drop a video to begin</span></div>
          <button class="run-button" type="button"><span class="run-spinner" aria-hidden="true"></span><span class="run-icon">${icon("play", 17)}</span><span class="run-label">Run model</span></button>
          <div class="live-stats" hidden><span class="fps-stat"><b>—</b> FPS</span><span class="payload-size" hidden><b>—</b> SVG BYTES</span><button type="button">${icon("restart", 15)} Restart</button></div>
        </div>
      </section>
      <section class="style-strip">
        <span class="step">STYLE</span>
        <div class="style-options">
          ${MODELS.map((model) => `<button type="button" data-model="${model.id}" aria-pressed="false"><strong>${model.label}</strong><small>${model.description}</small></button>`).join("")}
        </div>
        <label class="resolution-picker">
          <span>RESOLUTION</span>
          <select aria-label="Inference resolution">
            <option value="256">256 · Fast</option>
            <option value="384">384 · Balanced</option>
            <option value="512">512 · HQ</option>
          </select>
        </label>
        <label class="background-picker">
          <span>BACKGROUND</span>
          <input type="color" value="#00ff00" aria-label="Background color" disabled>
          <code>#00FF00</code>
        </label>
        <label class="segmentation-toggle">
          <input type="checkbox">
          <span><strong>PERSON MASK</strong><small>Full frame</small></span>
        </label>
        <label class="segmentation-picker">
          <span>MASK MODEL</span>
          <select aria-label="MediaPipe person segmentation model" disabled>
            <option value="selfie">Selfie · Soft</option>
            <option value="multiclass" selected>Multiclass · Soft</option>
            <option value="multiclass-category">Multiclass · Fast</option>
          </select>
        </label>
        <label class="mirror-toggle">
          <input type="checkbox">
          <span><strong>MIRROR INPUT</strong><small>Off</small></span>
        </label>
      </section>
      <section class="effect-controls cel-controls" hidden>
        <span class="step">CEL</span>
        <label><span>COLOR LEVELS <output>8</output></span><input class="cel-levels" type="range" min="2" max="25" step="1" value="8" aria-label="Cel shader color levels"></label>
        <label><span>EDGE SENSITIVITY <output>0.25</output></span><input class="cel-edge-threshold" type="range" min="0.05" max="0.8" step="0.01" value="0.25" aria-label="Cel shader edge sensitivity"></label>
        <label><span>EDGE THICKNESS <output>1</output></span><input class="cel-edge-thickness" type="range" min="1" max="3" step="1" value="1" aria-label="Cel shader edge thickness"></label>
      </section>
      <section class="effect-controls contour-controls" hidden>
        <span class="step">CONTOUR</span>
        <label><span>FILL MODE</span><select class="contour-fill-mode" aria-label="Contour fill mode"><option value="filled">Filled</option><option value="lines">Lines</option></select></label>
        <label><span>LEVELS <output>3</output></span><input class="contour-levels" type="range" min="2" max="12" step="1" value="3" aria-label="Contour levels"></label>
        <label><span>THICKNESS <output>0.50</output></span><input class="contour-thickness" type="range" min="0.25" max="4" step="0.25" value="0.5" aria-label="Contour thickness"></label>
      </section>
      <section class="effect-controls low-poly-controls" hidden>
        <span class="step">LOW POLY</span>
        <label><span>DETAIL <output>18</output></span><input class="low-poly-detail" type="range" min="6" max="32" step="1" value="18" aria-label="Low Poly detail"></label>
        <label><span>EDGE GUIDE <output>65%</output></span><input class="low-poly-edge" type="range" min="0" max="100" step="5" value="65" aria-label="Low Poly edge guidance"></label>
        <label><span>IRREGULARITY <output>35%</output></span><input class="low-poly-jitter" type="range" min="0" max="80" step="5" value="35" aria-label="Low Poly irregularity"></label>
      </section>
      <section class="effect-controls ervin-controls" hidden>
        <span class="step">ERVIN</span>
        <label><span>POINTS @256 <output>1200</output></span><input class="ervin-points" type="range" min="200" max="1600" step="100" value="1200" aria-label="Ervin feature points at 256 resolution"></label>
        <label><span>EDGE THRESHOLD <output>5%</output></span><input class="ervin-threshold" type="range" min="2" max="60" step="1" value="5" aria-label="Ervin edge threshold"></label>
        <label><span>BLUR <output>1.0</output></span><input class="ervin-blur" type="range" min="0" max="6" step="0.5" value="1" aria-label="Ervin blur radius"></label>
      </section>
      <section class="effect-controls vector-controls" hidden>
        <span class="step">VECTOR</span>
        <label><span>COLORS <output>12</output></span><input class="vector-colors" type="range" min="2" max="24" step="1" value="12" aria-label="Vector color count"></label>
        <label><span>TRACE DETAIL <output>128</output></span><input class="vector-detail" type="range" min="64" max="192" step="16" value="128" aria-label="Vector trace detail"></label>
        <label><span>SIMPLIFY <output>1.50</output></span><input class="vector-simplify" type="range" min="0" max="6" step="0.25" value="1.5" aria-label="Vector path simplification"></label>
        <label><span>BLUR <output>1</output></span><input class="vector-blur" type="range" min="0" max="5" step="1" value="1" aria-label="Vector selective blur radius"></label>
      </section>
    </main>
    <footer><span>CARTOONIZATION · LOCAL INFERENCE</span><span>Model use is subject to the upstream <a href="https://github.com/TachibanaYoshino/AnimeGANv3#-license" target="_blank" rel="noreferrer">AnimeGANv3</a> and <a href="https://github.com/SystemErrorWang/White-box-Cartoonization#license" target="_blank" rel="noreferrer">White-box</a> licenses</span></footer>
  </div>`;

const find = <T extends Element>(selector: string) => document.querySelector<T>(selector)!;
const dropZone = find<HTMLDivElement>(".drop-zone");
const fileInput = find<HTMLInputElement>("#file-input");
const exampleButton = find<HTMLButtonElement>(".example-button");
const webcamButton = find<HTMLButtonElement>(".webcam-button");
const sourceMessage = find<HTMLElement>(".source-message");
const workspace = find<HTMLElement>(".workspace");
const fileName = find<HTMLElement>("#file-name");
const removeButton = find<HTMLButtonElement>(".remove");
const video = find<HTMLVideoElement>(".source-tile video");
const canvas = find<HTMLCanvasElement>(".result-tile canvas");
const resultModel = find<HTMLElement>("#result-model");
const resultBackend = find<HTMLElement>("#result-backend");
const resultEmpty = find<HTMLElement>(".result-empty");
const cropGuideLabel = find<HTMLElement>(".crop-guide span");
const statusElement = find<HTMLElement>(".status");
const statusText = find<HTMLElement>("#status-text");
const runButton = find<HTMLButtonElement>(".run-button");
const runLabel = find<HTMLElement>(".run-label");
const liveStats = find<HTMLElement>(".live-stats");
const fpsText = find<HTMLElement>(".live-stats b");
const payloadSize = find<HTMLElement>(".payload-size");
const payloadText = find<HTMLElement>(".payload-size b");
const restartButton = find<HTMLButtonElement>(".live-stats button");
const styleButtons = [...document.querySelectorAll<HTMLButtonElement>("[data-model]")];
const resolutionInput = find<HTMLSelectElement>(".resolution-picker select");
const backgroundInput = find<HTMLInputElement>(".background-picker input");
const backgroundCode = find<HTMLElement>(".background-picker code");
const maskInput = find<HTMLInputElement>(".segmentation-toggle input");
const maskStatus = find<HTMLElement>(".segmentation-toggle small");
const segmentationModelInput = find<HTMLSelectElement>(".segmentation-picker select");
const mirrorInput = find<HTMLInputElement>(".mirror-toggle input");
const mirrorStatus = find<HTMLElement>(".mirror-toggle small");
const celControls = find<HTMLElement>(".cel-controls");
const celLevelsInput = find<HTMLInputElement>(".cel-levels");
const celEdgeThresholdInput = find<HTMLInputElement>(".cel-edge-threshold");
const celEdgeThicknessInput = find<HTMLInputElement>(".cel-edge-thickness");
const contourControls = find<HTMLElement>(".contour-controls");
const contourFillModeInput = find<HTMLSelectElement>(".contour-fill-mode");
const contourLevelsInput = find<HTMLInputElement>(".contour-levels");
const contourThicknessInput = find<HTMLInputElement>(".contour-thickness");
const lowPolyControls = find<HTMLElement>(".low-poly-controls");
const lowPolyDetailInput = find<HTMLInputElement>(".low-poly-detail");
const lowPolyEdgeInput = find<HTMLInputElement>(".low-poly-edge");
const lowPolyJitterInput = find<HTMLInputElement>(".low-poly-jitter");
const ervinControls = find<HTMLElement>(".ervin-controls");
const ervinPointsInput = find<HTMLInputElement>(".ervin-points");
const ervinThresholdInput = find<HTMLInputElement>(".ervin-threshold");
const ervinBlurInput = find<HTMLInputElement>(".ervin-blur");
const vectorControls = find<HTMLElement>(".vector-controls");
const vectorColorsInput = find<HTMLInputElement>(".vector-colors");
const vectorDetailInput = find<HTMLInputElement>(".vector-detail");
const vectorSimplifyInput = find<HTMLInputElement>(".vector-simplify");
const vectorBlurInput = find<HTMLInputElement>(".vector-blur");

const runner = new AnimeGanRunner();
let selectedModel = MODELS[0];
let videoUrl: string | undefined;
let ownsVideoUrl = false;
let cameraStream: MediaStream | undefined;
let processing = false;
let modelLoading = false;
let running = false;
let frameRequest: number | undefined;
let processingGeneration = 0;

function setStatus(status: Status, text: string) {
  statusElement.className = `status status-${status}`;
  statusText.textContent = text;
  runButton.disabled = status === "loading" || modelLoading;
}

function setModelLoading(value: boolean, model = selectedModel) {
  modelLoading = value;
  runButton.classList.toggle("loading", value);
  runButton.disabled = value;
  runLabel.textContent = value ? `Loading ${model.label}…` : "Run model";
}

function setProcessing(value: boolean) {
  processing = value;
  runButton.hidden = value;
  liveStats.hidden = !value;
  resultEmpty.hidden = value;
  video.controls = !value && !cameraStream;
  restartButton.hidden = Boolean(cameraStream);
}

function updateSelectedModel() {
  for (const button of styleButtons) {
    const active = button.dataset.model === selectedModel.id;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  resultModel.textContent = selectedModel.label.toUpperCase();
  celControls.hidden = selectedModel.engine !== "cel";
  contourControls.hidden = selectedModel.engine !== "contour";
  lowPolyControls.hidden = selectedModel.engine !== "lowpoly";
  ervinControls.hidden = selectedModel.engine !== "ervin";
  vectorControls.hidden = selectedModel.engine !== "vector";
  payloadSize.hidden = selectedModel.engine !== "vector";
}

function stopProcessing() {
  running = false;
  processingGeneration += 1;
  if (frameRequest !== undefined && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameRequest);
  frameRequest = undefined;
}

function releaseVideoUrl() {
  if (videoUrl && ownsVideoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = undefined;
  ownsVideoUrl = false;
}

function releaseCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
}

function hasVideoSource() {
  return Boolean(videoUrl || cameraStream);
}

function updateVideoClasses() {
  video.classList.toggle("portrait-input", video.videoHeight > video.videoWidth);
  video.classList.toggle("landscape-input", video.videoHeight <= video.videoWidth);
  video.classList.toggle("mirrored", mirrorInput.checked);
}

function clearVideo() {
  stopProcessing();
  setModelLoading(false);
  video.pause();
  releaseCamera();
  video.removeAttribute("src");
  video.load();
  releaseVideoUrl();
  fileInput.value = "";
  canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  workspace.hidden = true;
  dropZone.hidden = false;
  setProcessing(false);
  fpsText.textContent = "—";
  payloadText.textContent = "—";
  setStatus("idle", "Drop a video to begin");
}

function loadVideo(url: string, name: string, revokeOnRelease: boolean) {
  stopProcessing();
  setModelLoading(false);
  video.pause();
  releaseCamera();
  releaseVideoUrl();
  videoUrl = url;
  ownsVideoUrl = revokeOnRelease;
  fileName.textContent = name;
  video.loop = true;
  video.src = videoUrl;
  dropZone.hidden = true;
  workspace.hidden = false;
  fpsText.textContent = "—";
  payloadText.textContent = "—";
  setProcessing(false);
  setStatus("loading", "Preparing video…");
}

function acceptFile(file?: File) {
  if (!file || !file.type.startsWith("video/")) {
    setStatus("error", "Please choose a video file");
    return;
  }
  loadVideo(URL.createObjectURL(file), file.name, true);
}

async function startWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    sourceMessage.textContent = "This browser does not support webcam capture.";
    return;
  }
  sourceMessage.textContent = "Requesting camera access…";
  webcamButton.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    stopProcessing();
    setModelLoading(false);
    video.pause();
    releaseCamera();
    releaseVideoUrl();
    cameraStream = stream;
    video.removeAttribute("src");
    video.srcObject = stream;
    video.loop = false;
    fileName.textContent = "Webcam · live";
    dropZone.hidden = true;
    workspace.hidden = false;
    fpsText.textContent = "—";
    payloadText.textContent = "—";
    sourceMessage.textContent = "";
    setProcessing(false);
    setStatus("loading", "Starting webcam…");
    await video.play();
  } catch (error) {
    sourceMessage.textContent = error instanceof DOMException && error.name === "NotAllowedError"
      ? "Camera access was not allowed."
      : "Could not start the webcam.";
  } finally {
    webcamButton.disabled = false;
  }
}

function scheduleNextFrame() {
  if (!running || frameRequest !== undefined) return;
  if (video.requestVideoFrameCallback) {
    frameRequest = video.requestVideoFrameCallback(() => {
      frameRequest = undefined;
      void processNextFrame();
    });
  } else {
    frameRequest = requestAnimationFrame(() => {
      frameRequest = undefined;
      void processNextFrame();
    });
  }
}

async function processNextFrame() {
  if (!running || video.paused || video.ended) return;
  const started = performance.now();
  try {
    const result = await runner.render(video, video.videoWidth, video.videoHeight, canvas);
    const elapsed = performance.now() - started;
    fpsText.textContent = (1000 / elapsed).toFixed(1);
    payloadText.textContent = result.payloadBytes === undefined ? "—" : result.payloadBytes.toLocaleString();
    setStatus("ready", `Live · ${Math.round(elapsed)} ms/frame`);
  } catch (error) {
    stopProcessing();
    setStatus("error", error instanceof Error ? error.message : "Frame processing failed");
    return;
  }
  scheduleNextFrame();
}

async function beginProcessing() {
  const generation = ++processingGeneration;
  const model = selectedModel;
  setProcessing(false);
  setModelLoading(true, model);
  setStatus("loading", `Loading ${model.label} model…`);
  try {
    await runner.load(model, (loaded, total) => {
      const percent = total ? Math.round((loaded / total) * 100) : 0;
      setStatus("loading", `Loading ${model.label} model${percent ? ` · ${percent}%` : "…"}`);
    });
    if (generation !== processingGeneration) return;
    setModelLoading(false);
    const backendLabel = runner.backend === "webgpu" ? "WebGPU" : runner.backend === "canvas" ? "Canvas" : "WASM";
    resultBackend.textContent = backendLabel.toUpperCase();
    setStatus("ready", `${backendLabel} ready`);
    await video.play();
    if (generation !== processingGeneration) return;
    running = true;
    setProcessing(true);
    scheduleNextFrame();
  } catch (error) {
    if (generation !== processingGeneration) return;
    stopProcessing();
    setModelLoading(false);
    setProcessing(false);
    setStatus("error", error instanceof Error ? error.message : "Could not initialize the model");
  }
}

function chooseModel(model: AnimeModel) {
  if (model.id === selectedModel.id) return;
  stopProcessing();
  video.pause();
  selectedModel = model;
  setProcessing(false);
  fpsText.textContent = "—";
  payloadText.textContent = "—";
  resultBackend.textContent = maskInput.checked ? "GREEN SCREEN" : "FULL FRAME";
  canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  updateSelectedModel();
  if (hasVideoSource() && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    void beginProcessing();
  } else {
    setStatus(hasVideoSource() ? "loading" : "idle", hasVideoSource() ? "Waiting for the first video frame…" : "Drop a video to begin");
  }
}

dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  acceptFile(event.dataTransfer?.files[0]);
});
fileInput.addEventListener("change", () => acceptFile(fileInput.files?.[0]));
exampleButton.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  loadVideo(`${import.meta.env.BASE_URL}examples/black.mp4`, "black.mp4 · example", false);
});
webcamButton.addEventListener("click", () => void startWebcam());
removeButton.addEventListener("click", clearVideo);
runButton.addEventListener("click", () => void beginProcessing());
restartButton.addEventListener("click", () => { video.currentTime = 0; });
video.addEventListener("loadedmetadata", () => {
  updateVideoClasses();
  setStatus("loading", cameraStream ? "Starting webcam…" : "Decoding first frame…");
});
video.addEventListener("loadeddata", () => setStatus("ready", cameraStream ? "Webcam ready" : "Video ready"));
video.addEventListener("play", () => {
  if (!processing) return;
  running = true;
  scheduleNextFrame();
});
video.addEventListener("pause", () => { running = false; });
for (const button of styleButtons) {
  button.addEventListener("click", () => {
    const model = MODELS.find(({ id }) => id === button.dataset.model);
    if (model) chooseModel(model);
  });
}
backgroundInput.addEventListener("input", () => {
  runner.backgroundColor = backgroundInput.value;
  backgroundCode.textContent = backgroundInput.value.toUpperCase();
});
maskInput.addEventListener("change", () => {
  runner.useMediaPipe = maskInput.checked;
  backgroundInput.disabled = !maskInput.checked;
  segmentationModelInput.disabled = !maskInput.checked;
  maskStatus.textContent = maskInput.checked ? segmentationModelInput.selectedOptions[0].textContent : "Full frame";
  if (!processing) resultBackend.textContent = maskInput.checked ? "GREEN SCREEN" : "FULL FRAME";
});
segmentationModelInput.addEventListener("change", () => {
  runner.segmentationModel = segmentationModelInput.value as SegmentationModel;
  maskStatus.textContent = segmentationModelInput.selectedOptions[0].textContent;
  if (processing) setStatus("loading", `Switching to ${maskStatus.textContent}…`);
});
mirrorInput.addEventListener("change", () => {
  runner.mirrorInput = mirrorInput.checked;
  mirrorStatus.textContent = mirrorInput.checked ? "On" : "Off";
  updateVideoClasses();
});
resolutionInput.addEventListener("change", () => {
  const size = Number(resolutionInput.value);
  runner.inputSize = size;
  cropGuideLabel.textContent = `FULL HEIGHT · ${size}²`;
  if (processing) setStatus("ready", `${size}×${size} · applies on next frame`);
});
function bindRangeControl(input: HTMLInputElement, update: (value: number) => void, format = (value: string) => value) {
  input.addEventListener("input", () => {
    const value = Number(input.value);
    update(value);
    input.parentElement?.querySelector("output")?.replaceChildren(format(input.value));
  });
}
bindRangeControl(celLevelsInput, (value) => { runner.celLevels = value; });
bindRangeControl(celEdgeThresholdInput, (value) => { runner.celEdgeThreshold = value; });
bindRangeControl(celEdgeThicknessInput, (value) => { runner.celEdgeThickness = value; });
contourFillModeInput.addEventListener("change", () => {
  runner.contourLines = contourFillModeInput.value === "lines";
});
bindRangeControl(contourLevelsInput, (value) => { runner.contourLevels = value; });
bindRangeControl(contourThicknessInput, (value) => { runner.contourThickness = value; });
bindRangeControl(lowPolyDetailInput, (value) => { runner.lowPolyDetail = value; });
bindRangeControl(lowPolyEdgeInput, (value) => { runner.lowPolyEdgeGuidance = value / 100; }, (value) => `${value}%`);
bindRangeControl(lowPolyJitterInput, (value) => { runner.lowPolyJitter = value / 100; }, (value) => `${value}%`);
bindRangeControl(ervinPointsInput, (value) => { runner.ervinPoints = value; });
bindRangeControl(ervinThresholdInput, (value) => { runner.ervinThreshold = value / 100; }, (value) => `${value}%`);
bindRangeControl(ervinBlurInput, (value) => { runner.ervinBlur = value; }, (value) => Number(value).toFixed(1));
bindRangeControl(vectorColorsInput, (value) => { runner.vectorColors = value; });
bindRangeControl(vectorDetailInput, (value) => { runner.vectorDetail = value; });
bindRangeControl(vectorSimplifyInput, (value) => { runner.vectorSimplify = value; }, (value) => Number(value).toFixed(2));
bindRangeControl(vectorBlurInput, (value) => { runner.vectorBlur = value; });
window.addEventListener("beforeunload", () => {
  stopProcessing();
  releaseCamera();
  releaseVideoUrl();
  runner.dispose();
});

updateSelectedModel();
