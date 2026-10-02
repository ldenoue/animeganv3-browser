declare module "imagetracerjs" {
  export type ImageTracerColor = { r: number; g: number; b: number; a: number };

  export type ImageTracerSegment = {
    type: "L" | "Q";
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    x3?: number;
    y3?: number;
  };

  export type ImageTracerPath = {
    segments: ImageTracerSegment[];
    holechildren: number[];
    isholepath: boolean;
  };

  export type ImageTracerData = {
    layers: ImageTracerPath[][];
    palette: ImageTracerColor[];
    width: number;
    height: number;
  };

  export type ImageTracerOptions = {
    ltres?: number;
    qtres?: number;
    pathomit?: number;
    rightangleenhance?: boolean;
    colorsampling?: number;
    numberofcolors?: number;
    mincolorratio?: number;
    colorquantcycles?: number;
    layering?: number;
    blurradius?: number;
    blurdelta?: number;
  };

  const ImageTracer: {
    imagedataToTracedata(
      image: { width: number; height: number; data: Uint8ClampedArray },
      options?: ImageTracerOptions,
    ): ImageTracerData;
  };

  export default ImageTracer;
}
