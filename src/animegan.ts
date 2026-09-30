import type { AnimeModel } from "./models";

export type Backend = "webgpu" | "wasm" | "canvas";

type Reply =
  | { type: "progress"; received: number; total: number }
  | { type: "loaded"; requestId: number; backend: Backend }
  | { type: "frame"; requestId: number; pixels: ArrayBuffer; width: number; height: number; sourceWidth: number; sourceHeight: number }
  | { type: "error"; requestId: number; message: string };

export class AnimeGanRunner {
  private worker?: Worker;
  private pending = new Map<number, { resolve: (reply: Reply) => void; reject: (error: Error) => void }>();
  private requestId = 0;
  private progress?: (received: number, total: number) => void;
  private frameCanvas = document.createElement("canvas");
  private frameContext = this.frameCanvas.getContext("2d")!;
  private sourceCanvas = document.createElement("canvas");
  private sourceContext = this.sourceCanvas.getContext("2d")!;
  backend: Backend = "wasm";
  model?: AnimeModel;
  backgroundColor = "#00ff00";
  useMediaPipe = false;

  private ensureWorker() {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL("./animegan.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }: MessageEvent<Reply>) => {
      if (data.type === "progress") {
        this.progress?.(data.received, data.total);
        return;
      }
      const pending = this.pending.get(data.requestId);
      if (!pending) return;
      this.pending.delete(data.requestId);
      if (data.type === "error") pending.reject(new Error(data.message));
      else pending.resolve(data);
    };
    worker.onerror = ({ message }) => {
      const error = new Error(message || "Inference worker stopped");
      this.pending.forEach(({ reject }) => reject(error));
      this.pending.clear();
    };
    this.worker = worker;
    return worker;
  }

  private send(message: object, transfer: Transferable[] = []) {
    const requestId = ++this.requestId;
    const reply = new Promise<Reply>((resolve, reject) => this.pending.set(requestId, { resolve, reject }));
    this.ensureWorker().postMessage({ ...message, requestId }, transfer);
    return reply;
  }

  dispose() {
    this.worker?.terminate();
    this.worker = undefined;
    const error = new Error("Inference worker stopped");
    this.pending.forEach(({ reject }) => reject(error));
    this.pending.clear();
    this.progress = undefined;
    this.model = undefined;
  }

  async load(model: AnimeModel, onProgress?: (received: number, total: number) => void) {
    if (this.model?.id === model.id) return;
    this.progress = onProgress;
    const reply = await this.send({
      type: "load",
      url: model.url,
      stride: model.stride ?? 8,
      engine: model.engine ?? "animegan",
      colorOrder: model.colorOrder ?? "rgb",
    });
    if (reply.type !== "loaded") throw new Error("Unexpected model response");
    this.backend = reply.backend;
    this.model = model;
    this.progress = undefined;
  }

  async render(source: CanvasImageSource, sourceWidth: number, sourceHeight: number, output: HTMLCanvasElement) {
    if (!this.model) throw new Error("Model is not loaded");
    if (source instanceof HTMLVideoElement && source.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      throw new Error("Waiting for the first decoded video frame");
    }
    this.sourceCanvas.width = sourceWidth;
    this.sourceCanvas.height = sourceHeight;
    this.sourceContext.drawImage(source, 0, 0, sourceWidth, sourceHeight);
    const bitmap = await createImageBitmap(this.sourceCanvas);
    const background = this.backgroundColor.match(/[a-f\d]{2}/gi)?.map((channel) => Number.parseInt(channel, 16)) ?? [0, 255, 0];
    const reply = await this.send({ type: "render", bitmap, sourceWidth, sourceHeight, background, useMediaPipe: this.useMediaPipe }, [bitmap]);
    if (reply.type !== "frame") throw new Error("Unexpected frame response");

    this.frameCanvas.width = reply.width;
    this.frameCanvas.height = reply.height;
    this.frameContext.putImageData(new ImageData(new Uint8ClampedArray(reply.pixels), reply.width, reply.height), 0, 0);
    output.width = reply.sourceWidth;
    output.height = reply.sourceHeight;
    output.getContext("2d")!.drawImage(this.frameCanvas, 0, 0, reply.sourceWidth, reply.sourceHeight);
  }
}
