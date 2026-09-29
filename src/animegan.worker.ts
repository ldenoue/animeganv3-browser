/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import { ImageSegmenter } from "@mediapipe/tasks-vision";
import visionWasmLoader from "./vendor/mediapipe/vision_wasm_module_internal.js?url";
import visionWasmBinary from "./vendor/mediapipe/vision_wasm_module_internal.wasm?url";
import { vectorizeEdges, vectorizeMaskContours } from "./vectorize-edges";

type LoadMessage = { type: "load"; requestId: number; url?: string; stride: 8 | 16; engine: "animegan" | "palette" };
type RenderMessage = { type: "render"; requestId: number; bitmap: ImageBitmap; sourceWidth: number; sourceHeight: number; background: number[]; useMediaPipe: boolean };

let session: ort.InferenceSession | undefined;
let segmenter: ImageSegmenter | undefined;
let backend: "webgpu" | "wasm" | "canvas" = "wasm";
let currentEngine: "animegan" | "palette" = "animegan";
let modelStride: 8 | 16 = 8;
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext("2d", { willReadFrequently: true })!;
const maskCanvas = new OffscreenCanvas(1, 1);
const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })!;

function send(message: object, transfer: Transferable[] = []) {
  self.postMessage(message, { transfer });
}

async function ensureSegmenter() {
  if (segmenter) return;
  segmenter = await ImageSegmenter.createFromOptions({
    wasmLoaderPath: visionWasmLoader,
    wasmBinaryPath: visionWasmBinary,
  }, {
    baseOptions: {
      modelAssetPath: "/models/selfie_multiclass_256x256.tflite",
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  });
}

async function loadModel({ requestId, url, stride, engine }: LoadMessage) {
  currentEngine = engine;
  if (engine === "palette") {
    backend = "canvas";
    send({ type: "loaded", requestId, backend });
    return;
  }
  if (!url) throw new Error("AnimeGAN model URL is missing");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load model (${response.status})`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body?.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  if (reader) {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      send({ type: "progress", received, total });
    }
  } else {
    const bytes = new Uint8Array(await response.arrayBuffer());
    chunks.push(bytes);
    received = bytes.length;
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }

  const hasWebGpu = "gpu" in navigator;
  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: hasWebGpu ? ["webgpu"] : ["wasm"],
      graphOptimizationLevel: "all",
    });
    backend = hasWebGpu ? "webgpu" : "wasm";
  } catch (error) {
    if (!hasWebGpu) throw error;
    console.warn("WebGPU setup failed; falling back to WASM.", error);
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    backend = "wasm";
  }
  modelStride = stride;
  send({ type: "loaded", requestId, backend });
}

function buildPalette(data: Uint8ClampedArray, count = 12) {
  const colors: number[][] = [];
  for (let i = 0; i < data.length; i += 4) colors.push([data[i], data[i + 1], data[i + 2]]);
  colors.sort((a, b) => (a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2]));
  const palette = Array.from({ length: count }, (_, index) => [...colors[Math.floor((index + 0.5) * colors.length / count)]]);
  for (let iteration = 0; iteration < 6; iteration += 1) {
    const sums = Array.from({ length: count }, () => [0, 0, 0, 0]);
    for (const color of colors) {
      let closest = 0;
      let best = Infinity;
      for (let index = 0; index < palette.length; index += 1) {
        const dr = color[0] - palette[index][0];
        const dg = color[1] - palette[index][1];
        const db = color[2] - palette[index][2];
        const distance = dr * dr + dg * dg + db * db;
        if (distance < best) { best = distance; closest = index; }
      }
      sums[closest][0] += color[0]; sums[closest][1] += color[1]; sums[closest][2] += color[2]; sums[closest][3] += 1;
    }
    sums.forEach((sum, index) => {
      if (sum[3]) palette[index] = [sum[0] / sum[3], sum[1] / sum[3], sum[2] / sum[3]];
    });
  }
  return palette;
}

function paletteFrame(original: ImageData, alpha: Uint8ClampedArray, backgroundColor: number[], drawSilhouette: boolean) {
  const blurred = new OffscreenCanvas(256, 256);
  const blurredContext = blurred.getContext("2d")!;
  // Enough pre-blur to merge video noise without erasing facial structure.
  blurredContext.filter = "blur(2px)";
  blurredContext.drawImage(canvas, 0, 0);
  const low = new OffscreenCanvas(32, 32);
  const lowContext = low.getContext("2d", { willReadFrequently: true })!;
  lowContext.imageSmoothingEnabled = true;
  lowContext.drawImage(blurred, 0, 0, 32, 32);
  const lowData = lowContext.getImageData(0, 0, 32, 32).data;
  const palette = buildPalette(lowData);
  const quantized = new ImageData(32, 32);
  for (let pixel = 0; pixel < lowData.length; pixel += 4) {
    let selected = palette[0];
    let best = Infinity;
    for (const color of palette) {
      const dr = lowData[pixel] - color[0]; const dg = lowData[pixel + 1] - color[1]; const db = lowData[pixel + 2] - color[2];
      const distance = dr * dr + dg * dg + db * db;
      if (distance < best) { best = distance; selected = color; }
    }
    quantized.data[pixel] = selected[0];
    quantized.data[pixel + 1] = selected[1];
    quantized.data[pixel + 2] = selected[2];
    quantized.data[pixel + 3] = 255;
  }
  lowContext.putImageData(quantized, 0, 0);

  // Preserve the small palette while avoiding hard 8x8 pixel blocks.
  const smooth = new OffscreenCanvas(256, 256);
  const smoothContext = smooth.getContext("2d", { willReadFrequently: true })!;
  smoothContext.imageSmoothingEnabled = true;
  smoothContext.imageSmoothingQuality = "high";
  smoothContext.drawImage(low, 0, 0, 256, 256);
  const smoothData = smoothContext.getImageData(0, 0, 256, 256).data;
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  const output = new ImageData(256, 256);
  for (let pixel = 0; pixel < output.data.length; pixel += 4) {
    const mask = alpha[pixel] / 255;
    let selected = palette[0];
    let best = Infinity;
    for (const color of palette) {
      const dr = smoothData[pixel] - color[0];
      const dg = smoothData[pixel + 1] - color[1];
      const db = smoothData[pixel + 2] - color[2];
      const distance = dr * dr + dg * dg + db * db;
      if (distance < best) { best = distance; selected = color; }
    }
    // Re-snap interpolated pixels to a palette color. A small interpolated
    // contribution keeps region boundaries antialiased instead of blocky.
    const red = selected[0] * .9 + smoothData[pixel] * .1;
    const green = selected[1] * .9 + smoothData[pixel + 1] * .1;
    const blue = selected[2] * .9 + smoothData[pixel + 2] * .1;
    output.data[pixel] = red * mask + backgroundRed * (1 - mask);
    output.data[pixel + 1] = green * mask + backgroundGreen * (1 - mask);
    output.data[pixel + 2] = blue * mask + backgroundBlue * (1 - mask);
    output.data[pixel + 3] = 255;
  }

  smoothContext.putImageData(output, 0, 0);
  smoothContext.lineCap = "round";
  smoothContext.lineJoin = "round";
  const drawPaths = (paths: Array<{ closed: boolean; points: Array<{ x: number; y: number }> }>, color: string, width: number) => {
    smoothContext.strokeStyle = color;
    smoothContext.lineWidth = width;
    for (const path of paths) {
      if (path.points.length < 2) continue;
      smoothContext.beginPath();
      smoothContext.moveTo(path.points[0].x, path.points[0].y);
      for (let point = 1; point < path.points.length; point += 1) {
        smoothContext.lineTo(path.points[point].x, path.points[point].y);
      }
      if (path.closed) smoothContext.closePath();
      smoothContext.stroke();
    }
  };

  const internalEdges = vectorizeEdges(original.data, alpha, 256, 256, 50, 128, "canny", "perceptual");
  const pathLength = (path: { points: Array<{ x: number; y: number }> }) => path.points.slice(1).reduce((length, point, index) => {
    const previous = path.points[index];
    return length + Math.hypot(point.x - previous.x, point.y - previous.y);
  }, 0);
  const structuralEdges = internalEdges.paths
    .map((path: typeof internalEdges.paths[number]) => ({ path, length: pathLength(path) }))
    .filter(({ path, length }: { path: typeof internalEdges.paths[number]; length: number }) => length >= 14 && (path.strength ?? 0) >= 32)
    .sort((first: { path: typeof internalEdges.paths[number]; length: number }, second: { path: typeof internalEdges.paths[number]; length: number }) => (
      second.length * (second.path.strength ?? 0) - first.length * (first.path.strength ?? 0)
    ))
    .slice(0, 28)
    .map(({ path }: { path: typeof internalEdges.paths[number] }) => path);
  drawPaths(structuralEdges, "rgba(35, 25, 21, 0.62)", 1.15);
  if (drawSilhouette) {
    const silhouette = vectorizeMaskContours(alpha, 256, 256, 82, 128);
    drawPaths(silhouette.paths, "rgba(30, 22, 19, 0.84)", 1.7);
  }
  return smoothContext.getImageData(0, 0, 256, 256).data;
}

async function renderFrame(message: RenderMessage) {
  if (currentEngine === "animegan" && !session) throw new Error("Model is not loaded");
  const width = 256;
  const height = 256;
  canvas.width = width;
  canvas.height = height;
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  if (message.sourceWidth >= message.sourceHeight) {
    const sourceSize = message.sourceHeight;
    const sourceX = (message.sourceWidth - sourceSize) / 2;
    context.drawImage(message.bitmap, sourceX, 0, sourceSize, sourceSize, 0, 0, width, height);
  } else {
    const destinationWidth = message.sourceWidth * (height / message.sourceHeight);
    const destinationX = (width - destinationWidth) / 2;
    context.drawImage(
      message.bitmap,
      0,
      0,
      message.sourceWidth,
      message.sourceHeight,
      destinationX,
      0,
      destinationWidth,
      height,
    );
  }
  message.bitmap.close();
  const original = context.getImageData(0, 0, width, height);

  let alpha: Uint8ClampedArray;
  if (message.useMediaPipe) {
    await ensureSegmenter();
    const segmentation = segmenter!.segment(canvas);
    const backgroundMask = segmentation.confidenceMasks?.[0];
    if (!backgroundMask) throw new Error("MediaPipe did not return a person mask");
    const background = backgroundMask.getAsFloat32Array();
    const maskPixels = new Uint8ClampedArray(backgroundMask.width * backgroundMask.height * 4);
    for (let i = 0, out = 0; i < background.length; i += 1) {
      const person = Math.max(0, Math.min(1, (1 - background[i] - 0.08) / 0.84));
      const value = Math.round(person * person * (3 - 2 * person) * 255);
      maskPixels[out++] = value; maskPixels[out++] = value; maskPixels[out++] = value; maskPixels[out++] = 255;
    }
    maskCanvas.width = backgroundMask.width;
    maskCanvas.height = backgroundMask.height;
    maskContext.putImageData(new ImageData(maskPixels, backgroundMask.width, backgroundMask.height), 0, 0);
    backgroundMask.close();
    segmentation.confidenceMasks?.slice(1).forEach((mask) => mask.close());
    const scaledMask = new OffscreenCanvas(width, height);
    const scaledMaskContext = scaledMask.getContext("2d", { willReadFrequently: true })!;
    scaledMaskContext.imageSmoothingEnabled = true;
    scaledMaskContext.drawImage(maskCanvas, 0, 0, width, height);
    alpha = scaledMaskContext.getImageData(0, 0, width, height).data;
  } else {
    alpha = new Uint8ClampedArray(width * height * 4);
    alpha.fill(255);
  }
  const rgba = original.data;
  const [backgroundRed, backgroundGreen, backgroundBlue] = message.background;
  if (currentEngine === "palette") {
    const pixels = paletteFrame(original, alpha, message.background, message.useMediaPipe);
    send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
    return;
  }
  const rgb = new Float32Array(width * height * 3);
  for (let pixel = 0, tensor = 0; pixel < rgba.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    rgb[tensor++] = (rgba[pixel] * mix + backgroundRed * (1 - mix)) / 127.5 - 1;
    rgb[tensor++] = (rgba[pixel + 1] * mix + backgroundGreen * (1 - mix)) / 127.5 - 1;
    rgb[tensor++] = (rgba[pixel + 2] * mix + backgroundBlue * (1 - mix)) / 127.5 - 1;
  }
  const animeSession = session!;
  const results = await animeSession.run({ [animeSession.inputNames[0]]: new ort.Tensor("float32", rgb, [1, height, width, 3]) });
  const data = results[animeSession.outputNames[0]].data as Float32Array;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, out = 0; i < data.length; i += 3) {
    pixels[out++] = Math.max(0, Math.min(255, (data[i] + 1) * 127.5));
    pixels[out++] = Math.max(0, Math.min(255, (data[i + 1] + 1) * 127.5));
    pixels[out++] = Math.max(0, Math.min(255, (data[i + 2] + 1) * 127.5));
    pixels[out++] = 255;
  }
  send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
}

self.onmessage = async ({ data }: MessageEvent<LoadMessage | RenderMessage>) => {
  try {
    if (data.type === "load") await loadModel(data);
    else await renderFrame(data);
  } catch (error) {
    if (data.type === "render") data.bitmap.close();
    send({ type: "error", requestId: data.requestId, message: error instanceof Error ? error.message : String(error) });
  }
};
