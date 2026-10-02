/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import { ImageSegmenter } from "@mediapipe/tasks-vision";
import Delaunator from "delaunator";
import ImageTracer, { type ImageTracerPath } from "imagetracerjs";
import simplify from "simplify-js";
import visionWasmLoader from "./vendor/mediapipe/vision_wasm_module_internal.js?url";
import visionWasmBinary from "./vendor/mediapipe/vision_wasm_module_internal.wasm?url";
import { vectorizeEdges, vectorizeMaskContours } from "./vectorize-edges";

type LoadMessage = {
  type: "load";
  requestId: number;
  url?: string;
  stride: 8 | 16;
  engine: "animegan" | "palette" | "cel" | "contour" | "lowpoly" | "ervin" | "vector";
  colorOrder: "rgb" | "bgr";
};
type RenderMessage = {
  type: "render";
  requestId: number;
  bitmap: ImageBitmap;
  sourceWidth: number;
  sourceHeight: number;
  background: number[];
  useMediaPipe: boolean;
  celLevels: number;
  celEdgeThreshold: number;
  celEdgeThickness: number;
  contourLines: boolean;
  contourLevels: number;
  contourThickness: number;
  lowPolyDetail: number;
  lowPolyEdgeGuidance: number;
  lowPolyJitter: number;
  ervinPoints: number;
  ervinThreshold: number;
  ervinBlur: number;
  vectorColors: number;
  vectorDetail: number;
  vectorSimplify: number;
  vectorBlur: number;
};

let session: ort.InferenceSession | undefined;
let segmenter: ImageSegmenter | undefined;
let backend: "webgpu" | "wasm" | "canvas" = "wasm";
let currentEngine: "animegan" | "palette" | "cel" | "contour" | "lowpoly" | "ervin" | "vector" = "animegan";
let modelColorOrder: "rgb" | "bgr" = "rgb";
let modelStride: 8 | 16 = 8;
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext("2d", { willReadFrequently: true })!;
const maskCanvas = new OffscreenCanvas(1, 1);
const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })!;
const lowPolyCanvas = new OffscreenCanvas(1, 1);
const lowPolyContext = lowPolyCanvas.getContext("2d", { willReadFrequently: true })!;
const ervinSourceCanvas = new OffscreenCanvas(1, 1);
const ervinSourceContext = ervinSourceCanvas.getContext("2d", { willReadFrequently: true })!;
const ervinBlurCanvas = new OffscreenCanvas(1, 1);
const ervinBlurContext = ervinBlurCanvas.getContext("2d", { willReadFrequently: true })!;
const vectorSourceCanvas = new OffscreenCanvas(1, 1);
const vectorSourceContext = vectorSourceCanvas.getContext("2d", { willReadFrequently: true })!;
const vectorTraceCanvas = new OffscreenCanvas(1, 1);
const vectorTraceContext = vectorTraceCanvas.getContext("2d", { willReadFrequently: true })!;

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
      modelAssetPath: `${import.meta.env.BASE_URL}models/selfie_multiclass_256x256.tflite`,
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  });
}

async function loadModel({ requestId, url, stride, engine, colorOrder }: LoadMessage) {
  await session?.release();
  session = undefined;
  currentEngine = engine;
  modelColorOrder = colorOrder;
  if (engine !== "animegan") {
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
  const width = original.width;
  const height = original.height;
  const blurred = new OffscreenCanvas(width, height);
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
  const smooth = new OffscreenCanvas(width, height);
  const smoothContext = smooth.getContext("2d", { willReadFrequently: true })!;
  smoothContext.imageSmoothingEnabled = true;
  smoothContext.imageSmoothingQuality = "high";
  smoothContext.drawImage(low, 0, 0, width, height);
  const smoothData = smoothContext.getImageData(0, 0, width, height).data;
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  const output = new ImageData(width, height);
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

  const internalEdges = vectorizeEdges(original.data, alpha, width, height, 50, 128, "canny", "perceptual");
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
    const silhouette = vectorizeMaskContours(alpha, width, height, 82, 128);
    drawPaths(silhouette.paths, "rgba(30, 22, 19, 0.84)", 1.7);
  }
  return smoothContext.getImageData(0, 0, width, height).data;
}

function celShaderFrame(
  original: ImageData,
  alpha: Uint8ClampedArray,
  backgroundColor: number[],
  levels: number,
  edgeThreshold: number,
  edgeThickness: number,
) {
  const { width, height } = original;
  const composited = new Uint8ClampedArray(original.data.length);
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  for (let pixel = 0; pixel < composited.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    composited[pixel] = original.data[pixel] * mix + backgroundRed * (1 - mix);
    composited[pixel + 1] = original.data[pixel + 1] * mix + backgroundGreen * (1 - mix);
    composited[pixel + 2] = original.data[pixel + 2] * mix + backgroundBlue * (1 - mix);
    composited[pixel + 3] = 255;
  }

  const luminance = (x: number, y: number) => {
    const clampedX = Math.max(0, Math.min(width - 1, x));
    const clampedY = Math.max(0, Math.min(height - 1, y));
    const pixel = (clampedY * width + clampedX) * 4;
    return (composited[pixel] * 0.299 + composited[pixel + 1] * 0.587 + composited[pixel + 2] * 0.114) / 255;
  };
  const output = new Uint8ClampedArray(composited.length);
  const thickness = Math.max(1, Math.round(edgeThickness));
  const quantizationLevels = Math.max(2, Math.round(levels));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const north = luminance(x, y + thickness);
      const south = luminance(x, y - thickness);
      const east = luminance(x + thickness, y);
      const west = luminance(x - thickness, y);
      const northEast = luminance(x + thickness, y + thickness);
      const northWest = luminance(x - thickness, y + thickness);
      const southEast = luminance(x + thickness, y - thickness);
      const southWest = luminance(x - thickness, y - thickness);
      const gradientX = -northWest + northEast - 2 * west + 2 * east - southWest + southEast;
      const gradientY = -northWest - 2 * north - northEast + southWest + 2 * south + southEast;
      const edgeFactor = Math.hypot(gradientX, gradientY) > edgeThreshold ? 0 : 1;
      const pixel = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const normalized = composited[pixel + channel] / 255;
        output[pixel + channel] = Math.floor(normalized * quantizationLevels) / quantizationLevels * 255 * edgeFactor;
      }
      output[pixel + 3] = 255;
    }
  }
  return output;
}

// CPU port of filtr's MIT-licensed Contour fragment shader.
// Source: https://github.com/eurobuddha/filtr/blob/main/src/engine/shaders/effects.ts
function contourFrame(
  original: ImageData,
  alpha: Uint8ClampedArray,
  backgroundColor: number[],
  linesOnly: boolean,
  levels: number,
  thickness: number,
) {
  const { width, height } = original;
  const composited = new Uint8ClampedArray(original.data.length);
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  for (let pixel = 0; pixel < composited.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    composited[pixel] = original.data[pixel] * mix + backgroundRed * (1 - mix);
    composited[pixel + 1] = original.data[pixel + 1] * mix + backgroundGreen * (1 - mix);
    composited[pixel + 2] = original.data[pixel + 2] * mix + backgroundBlue * (1 - mix);
    composited[pixel + 3] = 255;
  }

  const sampleChannel = (x: number, y: number, channel: number) => {
    const clampedX = Math.max(0, Math.min(width - 1, x));
    const clampedY = Math.max(0, Math.min(height - 1, y));
    const x0 = Math.floor(clampedX);
    const y0 = Math.floor(clampedY);
    const x1 = Math.min(width - 1, x0 + 1);
    const y1 = Math.min(height - 1, y0 + 1);
    const fx = clampedX - x0;
    const fy = clampedY - y0;
    const topLeft = composited[(y0 * width + x0) * 4 + channel];
    const topRight = composited[(y0 * width + x1) * 4 + channel];
    const bottomLeft = composited[(y1 * width + x0) * 4 + channel];
    const bottomRight = composited[(y1 * width + x1) * 4 + channel];
    const top = topLeft + (topRight - topLeft) * fx;
    const bottom = bottomLeft + (bottomRight - bottomLeft) * fx;
    return top + (bottom - top) * fy;
  };
  const quantizationLevels = Math.max(2, Math.round(levels));
  const band = (x: number, y: number) => {
    const red = sampleChannel(x, y, 0);
    const green = sampleChannel(x, y, 1);
    const blue = sampleChannel(x, y, 2);
    const luminance = Math.min(0.999, (red * 0.299 + green * 0.587 + blue * 0.114) / 255);
    return Math.floor(luminance * quantizationLevels);
  };

  const output = new Uint8ClampedArray(composited.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixel = (y * width + x) * 4;
      if (linesOnly) {
        const centerBand = band(x, y);
        const edge = Math.abs(band(x + thickness, y) - centerBand)
          + Math.abs(band(x, y + thickness) - centerBand);
        const isLine = edge >= 0.5;
        output[pixel] = isLine ? 0 : composited[pixel];
        output[pixel + 1] = isLine ? 0 : composited[pixel + 1];
        output[pixel + 2] = isLine ? 0 : composited[pixel + 2];
      } else {
        const denominator = Math.max(quantizationLevels - 1, 1);
        for (let channel = 0; channel < 3; channel += 1) {
          const normalized = composited[pixel + channel] / 255;
          output[pixel + channel] = Math.min(255, Math.floor(normalized * quantizationLevels) / denominator * 255);
        }
      }
      output[pixel + 3] = 255;
    }
  }
  return output;
}

type MeshPoint = { x: number; y: number };
type MeshTriangle = { a: number; b: number; c: number; circleX: number; circleY: number; radiusSquared: number };

function makeMeshTriangle(points: MeshPoint[], a: number, b: number, c: number): MeshTriangle | undefined {
  const first = points[a];
  const second = points[b];
  const third = points[c];
  const divisor = 2 * (
    first.x * (second.y - third.y)
    + second.x * (third.y - first.y)
    + third.x * (first.y - second.y)
  );
  if (Math.abs(divisor) < 1e-6) return undefined;
  const firstLength = first.x * first.x + first.y * first.y;
  const secondLength = second.x * second.x + second.y * second.y;
  const thirdLength = third.x * third.x + third.y * third.y;
  const circleX = (
    firstLength * (second.y - third.y)
    + secondLength * (third.y - first.y)
    + thirdLength * (first.y - second.y)
  ) / divisor;
  const circleY = (
    firstLength * (third.x - second.x)
    + secondLength * (first.x - third.x)
    + thirdLength * (second.x - first.x)
  ) / divisor;
  const deltaX = first.x - circleX;
  const deltaY = first.y - circleY;
  return { a, b, c, circleX, circleY, radiusSquared: deltaX * deltaX + deltaY * deltaY };
}

function triangulate(points: MeshPoint[], width: number, height: number) {
  const pointCount = points.length;
  const extent = Math.max(width, height);
  const centerX = width / 2;
  const centerY = height / 2;
  const meshPoints = [
    ...points,
    { x: centerX - extent * 20, y: centerY - extent },
    { x: centerX, y: centerY + extent * 20 },
    { x: centerX + extent * 20, y: centerY - extent },
  ];
  const superTriangle = makeMeshTriangle(meshPoints, pointCount, pointCount + 1, pointCount + 2);
  let triangles = superTriangle ? [superTriangle] : [];

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const point = meshPoints[pointIndex];
    const badTriangles = triangles.filter((triangle) => {
      const deltaX = point.x - triangle.circleX;
      const deltaY = point.y - triangle.circleY;
      return deltaX * deltaX + deltaY * deltaY <= triangle.radiusSquared + 1e-5;
    });
    const badSet = new Set(badTriangles);
    const edges = new Map<string, { first: number; second: number; count: number }>();
    for (const triangle of badTriangles) {
      for (const [first, second] of [[triangle.a, triangle.b], [triangle.b, triangle.c], [triangle.c, triangle.a]]) {
        const key = first < second ? `${first}:${second}` : `${second}:${first}`;
        const existing = edges.get(key);
        if (existing) existing.count += 1;
        else edges.set(key, { first, second, count: 1 });
      }
    }
    triangles = triangles.filter((triangle) => !badSet.has(triangle));
    for (const edge of edges.values()) {
      if (edge.count !== 1) continue;
      const triangle = makeMeshTriangle(meshPoints, edge.first, edge.second, pointIndex);
      if (triangle) triangles.push(triangle);
    }
  }
  return triangles.filter(({ a, b, c }) => a < pointCount && b < pointCount && c < pointCount);
}

function lowPolyFrame(
  original: ImageData,
  alpha: Uint8ClampedArray,
  backgroundColor: number[],
  detail: number,
  edgeGuidance: number,
  jitter: number,
) {
  const { width, height } = original;
  const composited = new Uint8ClampedArray(original.data.length);
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  for (let pixel = 0; pixel < composited.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    composited[pixel] = original.data[pixel] * mix + backgroundRed * (1 - mix);
    composited[pixel + 1] = original.data[pixel + 1] * mix + backgroundGreen * (1 - mix);
    composited[pixel + 2] = original.data[pixel + 2] * mix + backgroundBlue * (1 - mix);
    composited[pixel + 3] = 255;
  }

  const sampleLuminance = (x: number, y: number) => {
    const sampleX = Math.max(0, Math.min(width - 1, Math.round(x)));
    const sampleY = Math.max(0, Math.min(height - 1, Math.round(y)));
    const pixel = (sampleY * width + sampleX) * 4;
    return (composited[pixel] * 0.299 + composited[pixel + 1] * 0.587 + composited[pixel + 2] * 0.114) / 255;
  };
  const divisions = Math.max(6, Math.min(32, Math.round(detail)));
  const cellWidth = width / divisions;
  const cellHeight = height / divisions;
  const gradientStep = Math.max(1, Math.min(cellWidth, cellHeight) * 0.12);
  const gradientAt = (x: number, y: number) => Math.hypot(
    sampleLuminance(x + gradientStep, y) - sampleLuminance(x - gradientStep, y),
    sampleLuminance(x, y + gradientStep) - sampleLuminance(x, y - gradientStep),
  );
  const hash = (x: number, y: number, seed: number) => {
    const value = Math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453;
    return value - Math.floor(value);
  };

  const points: MeshPoint[] = [];
  for (let row = 0; row <= divisions; row += 1) {
    for (let column = 0; column <= divisions; column += 1) {
      const boundary = row === 0 || column === 0 || row === divisions || column === divisions;
      const baseX = column / divisions * width;
      const baseY = row / divisions * height;
      if (boundary) {
        points.push({ x: baseX, y: baseY });
        continue;
      }
      const jitterX = (hash(column, row, 1) - 0.5) * cellWidth * jitter;
      const jitterY = (hash(column, row, 2) - 0.5) * cellHeight * jitter;
      let pointX = baseX + jitterX;
      let pointY = baseY + jitterY;
      if (edgeGuidance > 0) {
        let strongestX = pointX;
        let strongestY = pointY;
        let strongestGradient = 0;
        for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
          for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
            const candidateX = pointX + offsetX * cellWidth * 0.3;
            const candidateY = pointY + offsetY * cellHeight * 0.3;
            const strength = gradientAt(candidateX, candidateY);
            if (strength > strongestGradient) {
              strongestGradient = strength;
              strongestX = candidateX;
              strongestY = candidateY;
            }
          }
        }
        const attraction = Math.min(1, strongestGradient * 3) * edgeGuidance;
        pointX += (strongestX - pointX) * attraction;
        pointY += (strongestY - pointY) * attraction;
      }
      points.push({
        x: Math.max(baseX - cellWidth * 0.46, Math.min(baseX + cellWidth * 0.46, pointX)),
        y: Math.max(baseY - cellHeight * 0.46, Math.min(baseY + cellHeight * 0.46, pointY)),
      });
    }
  }

  const triangles = triangulate(points, width, height);
  lowPolyCanvas.width = width;
  lowPolyCanvas.height = height;
  lowPolyContext.clearRect(0, 0, width, height);
  lowPolyContext.lineJoin = "bevel";
  lowPolyContext.lineWidth = 0.8;
  for (const triangle of triangles) {
    const first = points[triangle.a];
    const second = points[triangle.b];
    const third = points[triangle.c];
    const centerX = Math.max(0, Math.min(width - 1, Math.round((first.x + second.x + third.x) / 3)));
    const centerY = Math.max(0, Math.min(height - 1, Math.round((first.y + second.y + third.y) / 3)));
    const centerPixel = (centerY * width + centerX) * 4;
    const color = `rgb(${composited[centerPixel]} ${composited[centerPixel + 1]} ${composited[centerPixel + 2]})`;
    lowPolyContext.fillStyle = color;
    lowPolyContext.strokeStyle = color;
    lowPolyContext.beginPath();
    lowPolyContext.moveTo(first.x, first.y);
    lowPolyContext.lineTo(second.x, second.y);
    lowPolyContext.lineTo(third.x, third.y);
    lowPolyContext.closePath();
    lowPolyContext.fill();
    lowPolyContext.stroke();
  }
  return lowPolyContext.getImageData(0, 0, width, height).data;
}

function ervinFrame(
  original: ImageData,
  alpha: Uint8ClampedArray,
  backgroundColor: number[],
  pointBudget: number,
  threshold: number,
  blurRadius: number,
) {
  const { width, height } = original;
  const composited = new Uint8ClampedArray(original.data.length);
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  for (let pixel = 0; pixel < composited.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    composited[pixel] = original.data[pixel] * mix + backgroundRed * (1 - mix);
    composited[pixel + 1] = original.data[pixel + 1] * mix + backgroundGreen * (1 - mix);
    composited[pixel + 2] = original.data[pixel + 2] * mix + backgroundBlue * (1 - mix);
    composited[pixel + 3] = 255;
  }

  // Ervin Szilagyi's pipeline starts with a small Gaussian blur before edge
  // detection. Canvas blur is GPU-accelerated where available and gives the
  // same useful noise suppression without shipping an image-processing stack.
  ervinSourceCanvas.width = width;
  ervinSourceCanvas.height = height;
  ervinSourceContext.putImageData(new ImageData(composited, width, height), 0, 0);
  ervinBlurCanvas.width = width;
  ervinBlurCanvas.height = height;
  ervinBlurContext.clearRect(0, 0, width, height);
  ervinBlurContext.filter = `blur(${Math.max(0, blurRadius) * width / 256}px)`;
  ervinBlurContext.drawImage(ervinSourceCanvas, 0, 0);
  ervinBlurContext.filter = "none";
  const blurred = ervinBlurContext.getImageData(0, 0, width, height).data;

  const luminance = new Float32Array(width * height);
  for (let pixel = 0, sample = 0; pixel < blurred.length; pixel += 4, sample += 1) {
    luminance[sample] = blurred[pixel] * 0.299 + blurred[pixel + 1] * 0.587 + blurred[pixel + 2] * 0.114;
  }

  // An eight-neighbour Laplacian mirrors the detector used for the article's
  // featured output. Values remain normalized so one threshold works at every
  // selectable render resolution.
  const edges = new Float32Array(width * height);
  let edgeCount = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const neighbours = (
        luminance[index - width - 1] + luminance[index - width] + luminance[index - width + 1]
        + luminance[index - 1] + luminance[index + 1]
        + luminance[index + width - 1] + luminance[index + width] + luminance[index + width + 1]
      );
      const strength = Math.min(1, Math.abs(luminance[index] * 8 - neighbours) / 255);
      edges[index] = strength;
      if (strength >= threshold) edgeCount += 1;
    }
  }

  // Scale the point cap with image area so a given setting produces similar
  // facet sizes at 256, 384 and 512. The deterministic scan-and-stride cap is
  // the same selection strategy used by the reference implementation and is
  // more temporally stable than random sampling for video.
  const resolutionScale = width * height / (256 * 256);
  const maxFeaturePoints = Math.max(80, Math.min(6400, Math.round(pointBudget * resolutionScale)));
  const stride = Math.max(1, edgeCount / maxFeaturePoints);
  const points: MeshPoint[] = [];
  const boundaryStep = Math.max(4, Math.sqrt(width * height / Math.max(maxFeaturePoints, 1)));
  for (let x = 0; x < width; x += boundaryStep) {
    points.push({ x, y: 0 }, { x, y: height - 1 });
  }
  for (let y = boundaryStep; y < height - boundaryStep / 2; y += boundaryStep) {
    points.push({ x: 0, y }, { x: width - 1, y });
  }
  points.push(
    { x: width - 1, y: 0 },
    { x: width - 1, y: height - 1 },
    { x: 0, y: height - 1 },
  );

  let qualifyingIndex = 0;
  let nextSelection = 0;
  let selectedFeaturePoints = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      if (edges[y * width + x] < threshold) continue;
      if (qualifyingIndex >= nextSelection && selectedFeaturePoints < maxFeaturePoints) {
        points.push({ x, y });
        selectedFeaturePoints += 1;
        nextSelection += stride;
      }
      qualifyingIndex += 1;
    }
  }

  const coordinates = new Float64Array(points.length * 2);
  points.forEach((point, index) => {
    coordinates[index * 2] = point.x;
    coordinates[index * 2 + 1] = point.y;
  });
  const triangles = new Delaunator(coordinates).triangles;

  lowPolyCanvas.width = width;
  lowPolyCanvas.height = height;
  lowPolyContext.clearRect(0, 0, width, height);
  lowPolyContext.lineJoin = "bevel";
  lowPolyContext.lineWidth = 0.8;
  for (let index = 0; index < triangles.length; index += 3) {
    const first = points[triangles[index]];
    const second = points[triangles[index + 1]];
    const third = points[triangles[index + 2]];
    const centerX = Math.max(0, Math.min(width - 1, Math.round((first.x + second.x + third.x) / 3)));
    const centerY = Math.max(0, Math.min(height - 1, Math.round((first.y + second.y + third.y) / 3)));
    const centerPixel = (centerY * width + centerX) * 4;
    const color = `rgb(${composited[centerPixel]} ${composited[centerPixel + 1]} ${composited[centerPixel + 2]})`;
    lowPolyContext.fillStyle = color;
    lowPolyContext.strokeStyle = color;
    lowPolyContext.beginPath();
    lowPolyContext.moveTo(first.x, first.y);
    lowPolyContext.lineTo(second.x, second.y);
    lowPolyContext.lineTo(third.x, third.y);
    lowPolyContext.closePath();
    lowPolyContext.fill();
    lowPolyContext.stroke();
  }
  return lowPolyContext.getImageData(0, 0, width, height).data;
}

function vectorPathPoints(path: ImageTracerPath, tolerance: number) {
  if (!path.segments.length) return [];
  const points: Array<{ x: number; y: number }> = [
    { x: path.segments[0].x1, y: path.segments[0].y1 },
  ];
  for (const segment of path.segments) {
    if (segment.type === "Q" && segment.x3 !== undefined && segment.y3 !== undefined) {
      points.push({
        x: segment.x1 * 0.25 + segment.x2 * 0.5 + segment.x3 * 0.25,
        y: segment.y1 * 0.25 + segment.y2 * 0.5 + segment.y3 * 0.25,
      });
      points.push({ x: segment.x3, y: segment.y3 });
    } else {
      points.push({ x: segment.x2, y: segment.y2 });
    }
  }
  return tolerance > 0 ? simplify(points, tolerance, false) : points;
}

function vectorFrame(
  original: ImageData,
  alpha: Uint8ClampedArray,
  backgroundColor: number[],
  colorCount: number,
  traceDetail: number,
  simplification: number,
  blurRadius: number,
) {
  const { width, height } = original;
  const composited = new Uint8ClampedArray(original.data.length);
  const [backgroundRed, backgroundGreen, backgroundBlue] = backgroundColor;
  for (let pixel = 0; pixel < composited.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    composited[pixel] = original.data[pixel] * mix + backgroundRed * (1 - mix);
    composited[pixel + 1] = original.data[pixel + 1] * mix + backgroundGreen * (1 - mix);
    composited[pixel + 2] = original.data[pixel + 2] * mix + backgroundBlue * (1 - mix);
    composited[pixel + 3] = 255;
  }

  const detail = Math.max(48, Math.min(192, Math.round(traceDetail)));
  vectorSourceCanvas.width = width;
  vectorSourceCanvas.height = height;
  vectorSourceContext.putImageData(new ImageData(composited, width, height), 0, 0);
  vectorTraceCanvas.width = detail;
  vectorTraceCanvas.height = detail;
  vectorTraceContext.imageSmoothingEnabled = true;
  vectorTraceContext.imageSmoothingQuality = "high";
  vectorTraceContext.clearRect(0, 0, detail, detail);
  vectorTraceContext.drawImage(vectorSourceCanvas, 0, 0, detail, detail);
  const traceImage = vectorTraceContext.getImageData(0, 0, detail, detail);

  const traced = ImageTracer.imagedataToTracedata(traceImage, {
    ltres: 1,
    qtres: 1,
    pathomit: Math.max(2, Math.round(detail / 48)),
    rightangleenhance: false,
    colorsampling: 2,
    numberofcolors: Math.max(2, Math.min(24, Math.round(colorCount))),
    mincolorratio: 0,
    colorquantcycles: 2,
    layering: 0,
    blurradius: Math.max(0, Math.min(5, Math.round(blurRadius))),
    blurdelta: 32,
  });

  lowPolyCanvas.width = width;
  lowPolyCanvas.height = height;
  const paletteCounts = new Uint32Array(traced.palette.length);
  for (let pixel = 0; pixel < traceImage.data.length; pixel += 4) {
    let closest = 0;
    let closestDistance = Infinity;
    traced.palette.forEach((color, colorIndex) => {
      const red = traceImage.data[pixel] - color.r;
      const green = traceImage.data[pixel + 1] - color.g;
      const blue = traceImage.data[pixel + 2] - color.b;
      const distance = red * red + green * green + blue * blue;
      if (distance < closestDistance) {
        closest = colorIndex;
        closestDistance = distance;
      }
    });
    paletteCounts[closest] += 1;
  }
  let dominantIndex = 0;
  for (let index = 1; index < paletteCounts.length; index += 1) {
    if (paletteCounts[index] > paletteCounts[dominantIndex]) dominantIndex = index;
  }
  const dominant = traced.palette[dominantIndex] ?? { r: 0, g: 0, b: 0, a: 255 };
  const colorHex = (color: { r: number; g: number; b: number }) => `#${[color.r, color.g, color.b]
    .map((channel) => Math.round(channel).toString(16).padStart(2, "0"))
    .join("")}`;
  const dominantHex = colorHex(dominant);
  lowPolyContext.fillStyle = `rgba(${dominant.r} ${dominant.g} ${dominant.b} / ${dominant.a / 255})`;
  lowPolyContext.fillRect(0, 0, width, height);
  const scaleX = width / traced.width;
  const scaleY = height / traced.height;
  const coordinate = (value: number) => String(Number(value.toFixed(1)));
  const pointCache = new Map<ImageTracerPath, Array<{ x: number; y: number }>>();
  const appendPath = (path: ImageTracerPath) => {
    let points = pointCache.get(path);
    if (!points) {
      points = vectorPathPoints(path, Math.max(0, simplification));
      pointCache.set(path, points);
    }
    if (points.length < 3) return "";
    lowPolyContext.moveTo(points[0].x * scaleX, points[0].y * scaleY);
    for (let index = 1; index < points.length; index += 1) {
      lowPolyContext.lineTo(points[index].x * scaleX, points[index].y * scaleY);
    }
    lowPolyContext.closePath();
    return `M${points.map((point) => `${coordinate(point.x)} ${coordinate(point.y)}`).join("L")}Z`;
  };

  const svgParts = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${traced.width} ${traced.height}">`,
    `<path fill="${dominantHex}" d="M0 0H${traced.width}V${traced.height}H0Z"/>`,
  ];
  traced.layers.forEach((layer, layerIndex) => {
    const color = traced.palette[layerIndex];
    if (!color) return;
    const paint = `rgba(${color.r} ${color.g} ${color.b} / ${color.a / 255})`;
    lowPolyContext.fillStyle = paint;
    layer.forEach((path) => {
      if (path.isholepath) return;
      lowPolyContext.beginPath();
      let pathData = appendPath(path);
      if (!pathData) return;
      for (const childIndex of path.holechildren) {
        const hole = layer[childIndex];
        if (hole) pathData += appendPath(hole);
      }
      lowPolyContext.fill("evenodd");
      const opacity = color.a < 255 ? ` fill-opacity="${Number((color.a / 255).toFixed(3))}"` : "";
      svgParts.push(`<path fill="${colorHex(color)}"${opacity} fill-rule="evenodd" d="${pathData}"/>`);
    });
  });
  svgParts.push("</svg>");
  const payloadBytes = new TextEncoder().encode(svgParts.join("")).byteLength;
  return { pixels: lowPolyContext.getImageData(0, 0, width, height).data, payloadBytes };
}

async function renderFrame(message: RenderMessage) {
  if (currentEngine === "animegan" && !session) throw new Error("Model is not loaded");
  const width = message.sourceWidth;
  const height = message.sourceHeight;
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
  if (currentEngine === "cel") {
    const pixels = celShaderFrame(
      original,
      alpha,
      message.background,
      message.celLevels,
      message.celEdgeThreshold,
      message.celEdgeThickness,
    );
    send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
    return;
  }
  if (currentEngine === "contour") {
    const pixels = contourFrame(
      original,
      alpha,
      message.background,
      message.contourLines,
      message.contourLevels,
      message.contourThickness,
    );
    send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
    return;
  }
  if (currentEngine === "lowpoly") {
    const pixels = lowPolyFrame(
      original,
      alpha,
      message.background,
      message.lowPolyDetail,
      message.lowPolyEdgeGuidance,
      message.lowPolyJitter,
    );
    send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
    return;
  }
  if (currentEngine === "ervin") {
    const pixels = ervinFrame(
      original,
      alpha,
      message.background,
      message.ervinPoints,
      message.ervinThreshold,
      message.ervinBlur,
    );
    send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
    return;
  }
  if (currentEngine === "vector") {
    const { pixels, payloadBytes } = vectorFrame(
      original,
      alpha,
      message.background,
      message.vectorColors,
      message.vectorDetail,
      message.vectorSimplify,
      message.vectorBlur,
    );
    send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height, payloadBytes }, [pixels.buffer]);
    return;
  }
  const rgb = new Float32Array(width * height * 3);
  for (let pixel = 0, tensor = 0; pixel < rgba.length; pixel += 4) {
    const mix = alpha[pixel] / 255;
    const red = rgba[pixel] * mix + backgroundRed * (1 - mix);
    const green = rgba[pixel + 1] * mix + backgroundGreen * (1 - mix);
    const blue = rgba[pixel + 2] * mix + backgroundBlue * (1 - mix);
    rgb[tensor++] = (modelColorOrder === "bgr" ? blue : red) / 127.5 - 1;
    rgb[tensor++] = green / 127.5 - 1;
    rgb[tensor++] = (modelColorOrder === "bgr" ? red : blue) / 127.5 - 1;
  }
  const animeSession = session!;
  const results = await animeSession.run({ [animeSession.inputNames[0]]: new ort.Tensor("float32", rgb, [1, height, width, 3]) });
  const data = results[animeSession.outputNames[0]].data as Float32Array;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, out = 0; i < data.length; i += 3) {
    const red = modelColorOrder === "bgr" ? data[i + 2] : data[i];
    const blue = modelColorOrder === "bgr" ? data[i] : data[i + 2];
    pixels[out++] = Math.max(0, Math.min(255, (red + 1) * 127.5));
    pixels[out++] = Math.max(0, Math.min(255, (data[i + 1] + 1) * 127.5));
    pixels[out++] = Math.max(0, Math.min(255, (blue + 1) * 127.5));
    pixels[out++] = 255;
  }
  send({ type: "frame", requestId: message.requestId, pixels: pixels.buffer, width, height, sourceWidth: width, sourceHeight: height }, [pixels.buffer]);
}

async function handleMessage(data: LoadMessage | RenderMessage) {
  try {
    if (data.type === "load") await loadModel(data);
    else await renderFrame(data);
  } catch (error) {
    if (data.type === "render") data.bitmap.close();
    send({ type: "error", requestId: data.requestId, message: error instanceof Error ? error.message : String(error) });
  }
}

let operationQueue = Promise.resolve();
self.onmessage = ({ data }: MessageEvent<LoadMessage | RenderMessage>) => {
  operationQueue = operationQueue.then(() => handleMessage(data));
};
