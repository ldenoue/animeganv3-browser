import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Code2, Film, Play, RotateCcw, Sparkles, Upload, X } from "lucide-react";
import { AnimeGanRunner, type Backend } from "./animegan";
import { MODELS } from "./models";
import "./styles.css";

type Status = "idle" | "loading" | "ready" | "error";

function App() {
  const [videoUrl, setVideoUrl] = useState<string>();
  const [fileName, setFileName] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [statusText, setStatusText] = useState("Drop a video to begin");
  const [backend, setBackend] = useState<Backend | null>(null);
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [dragging, setDragging] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [fps, setFps] = useState(0);
  const [isPortrait, setIsPortrait] = useState(false);
  const [backgroundColor, setBackgroundColor] = useState("#00ff00");
  const [useMediaPipe, setUseMediaPipe] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runnerRef = useRef(new AnimeGanRunner());
  const runningRef = useRef(false);
  const frameRequestRef = useRef<number | undefined>(undefined);

  const clearVideo = useCallback(() => {
    runningRef.current = false;
    if (frameRequestRef.current && videoRef.current?.cancelVideoFrameCallback) {
      videoRef.current.cancelVideoFrameCallback(frameRequestRef.current);
    }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(undefined);
    setFileName("");
    setStatus("idle");
    setStatusText("Drop a video to begin");
    setProcessing(false);
    setFps(0);
  }, [videoUrl]);

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);

  const acceptFile = useCallback((file?: File) => {
    if (!file || !file.type.startsWith("video/")) {
      setStatus("error");
      setStatusText("Please choose a video file");
      return;
    }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setStatus("loading");
    setStatusText("Preparing video…");
    setFps(0);
  }, [videoUrl]);

  const processNextFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!runningRef.current || !video || !canvas || video.paused || video.ended) return;
    const started = performance.now();
    try {
      await runnerRef.current.render(video, video.videoWidth, video.videoHeight, canvas);
      const elapsed = performance.now() - started;
      setFps(1000 / elapsed);
      setStatusText(`Live · ${Math.round(elapsed)} ms/frame`);
    } catch (error) {
      runningRef.current = false;
      setStatus("error");
      setStatusText(error instanceof Error ? error.message : "Frame processing failed");
      return;
    }
    if (runningRef.current && video.requestVideoFrameCallback) {
      frameRequestRef.current = video.requestVideoFrameCallback(() => void processNextFrame());
    } else if (runningRef.current) {
      requestAnimationFrame(() => void processNextFrame());
    }
  }, []);

  const beginProcessing = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    setProcessing(true);
    setStatus("loading");
    setStatusText(`Loading ${selectedModel.label} model…`);
    try {
      await runnerRef.current.load(selectedModel, (loaded, total) => {
        const percent = total ? Math.round((loaded / total) * 100) : 0;
        setStatusText(`Loading ${selectedModel.label} model${percent ? ` · ${percent}%` : "…"}`);
      });
      setBackend(runnerRef.current.backend);
      setStatus("ready");
      const backendLabel = runnerRef.current.backend === "webgpu" ? "WebGPU" : runnerRef.current.backend === "canvas" ? "Canvas" : "WASM";
      setStatusText(`${backendLabel} ready`);
      runningRef.current = true;
      await video.play();
    } catch (error) {
      setProcessing(false);
      setStatus("error");
      setStatusText(error instanceof Error ? error.message : "Could not initialize the model");
    }
  }, [selectedModel]);

  const chooseModel = (model: typeof MODELS[number]) => {
    if (model.id === selectedModel.id) return;
    runningRef.current = false;
    videoRef.current?.pause();
    setSelectedModel(model);
    setProcessing(false);
    setBackend(null);
    setFps(0);
    setStatus(videoUrl ? "ready" : "idle");
    setStatusText(videoUrl ? `${model.label} selected · ready to run` : "Drop a video to begin");
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  };

  const handlePlay = () => {
    if (!processing) return;
    const video = videoRef.current;
    if (!video) return;
    runningRef.current = true;
    if (video.requestVideoFrameCallback) {
      frameRequestRef.current = video.requestVideoFrameCallback(() => void processNextFrame());
    } else if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      requestAnimationFrame(() => void processNextFrame());
    } else {
      video.addEventListener("loadeddata", () => void processNextFrame(), { once: true });
    }
  };

  const handlePause = () => { runningRef.current = false; };

  return (
    <div className="app-shell">
      <header>
        <a className="brand" href="#" aria-label="Frame by Frame home"><span className="brand-mark"><Film size={19} /></span><span>FRAME <i>BY</i> FRAME</span></a>
        <div className="header-meta"><span className="privacy"><span /> Runs entirely on your device</span><a href="https://github.com/TachibanaYoshino/AnimeGANv3" target="_blank" rel="noreferrer"><Code2 size={18} /> Source</a></div>
      </header>

      <main>
        <section className="intro">
          <div className="eyebrow"><span>01</span> VIDEO TO ANIME</div>
          <h1>See your world<br /><em>drawn differently.</em></h1>
          <p>Drop in a video. MediaPipe places detected people on chroma green, then AnimeGANv3 redraws the complete frame locally.</p>
        </section>

        {!videoUrl ? (
          <label
            className={`drop-zone ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]); }}
          >
            <input type="file" accept="video/*" onChange={(event) => acceptFile(event.target.files?.[0])} />
            <span className="upload-icon"><Upload size={26} /></span>
            <strong>Drop your video here</strong>
            <span>or click to browse · MP4, WebM, MOV</span>
            <span className="choose-button">Choose a video</span>
          </label>
        ) : (
          <section className="workspace">
            <div className="workspace-head">
              <div><span className="step">02</span><strong>{fileName}</strong></div>
              <button className="remove" onClick={clearVideo} aria-label="Remove video"><X size={17} /> Replace</button>
            </div>
            <div className="tiles">
              <article className="tile source-tile">
                <div className="tile-label"><span>ORIGINAL</span><small>INPUT</small></div>
                <video ref={videoRef} className={isPortrait ? "portrait-input" : "landscape-input"} src={videoUrl} muted loop playsInline preload="auto" controls={!processing}
                  onLoadedMetadata={(event) => {
                    setIsPortrait(event.currentTarget.videoHeight > event.currentTarget.videoWidth);
                    setStatusText("Decoding first frame…");
                  }}
                  onLoadedData={() => {
                    setStatus("ready");
                    setStatusText("Video ready");
                  }}
                  onPlay={handlePlay} onPause={handlePause}
                />
                <div className="crop-guide"><span>FULL HEIGHT · 256²</span></div>
              </article>
              <article className="tile result-tile">
                <div className="tile-label"><span>{selectedModel.label.toUpperCase()}</span><small>{backend?.toUpperCase() ?? (useMediaPipe ? "GREEN SCREEN" : "FULL FRAME")}</small></div>
                <canvas ref={canvasRef} />
                {!processing && <div className="result-empty"><Sparkles size={30} /><span>Your anime frames<br />will appear here</span></div>}
              </article>
            </div>
            <div className="control-bar">
              <div className={`status status-${status}`}><span />{statusText}</div>
              {!processing ? (
                <button className="run-button" disabled={status === "loading"} onClick={() => void beginProcessing()}><Play size={17} fill="currentColor" /> Run AnimeGAN</button>
              ) : (
                <div className="live-stats"><b>{fps ? fps.toFixed(1) : "—"}</b> FPS <button onClick={() => { videoRef.current!.currentTime = 0; }}><RotateCcw size={15} /> Restart</button></div>
              )}
            </div>
          </section>
        )}

        <section className="style-strip">
          <span className="step">STYLE</span>
          <div className="style-options">
            {MODELS.map((model) => (
              <button
                className={model.id === selectedModel.id ? "active" : ""}
                key={model.id}
                onClick={() => chooseModel(model)}
                aria-pressed={model.id === selectedModel.id}
              >
                <strong>{model.label}</strong>
                <small>{model.description}</small>
              </button>
            ))}
          </div>
          <label className="background-picker">
            <span>BACKGROUND</span>
            <input
              type="color"
              value={backgroundColor}
              disabled={!useMediaPipe}
              onChange={(event) => {
                setBackgroundColor(event.target.value);
                runnerRef.current.backgroundColor = event.target.value;
              }}
              aria-label="Background color"
            />
            <code>{backgroundColor.toUpperCase()}</code>
          </label>
          <label className="segmentation-toggle">
            <input
              type="checkbox"
              checked={useMediaPipe}
              onChange={(event) => {
                setUseMediaPipe(event.target.checked);
                runnerRef.current.useMediaPipe = event.target.checked;
              }}
            />
            <span><strong>PERSON MASK</strong><small>{useMediaPipe ? "MediaPipe on" : "Full frame"}</small></span>
          </label>
        </section>
      </main>
      <footer><span>ANIMEGANV3 · LOCAL INFERENCE</span><span>Model use is subject to the <a href="https://github.com/TachibanaYoshino/AnimeGANv3#-license" target="_blank" rel="noreferrer">AnimeGANv3 license</a></span></footer>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
