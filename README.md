# Frame by Frame

Local, in-browser video stylization using a framework-free HTML and TypeScript interface, optional MediaPipe multiclass segmentation, and ONNX Runtime Web. WebGPU is selected when available, with a WebAssembly fallback. Frames are mapped to a selectable 256×256, 384×384, or 512×512 inference size while preserving their full height: landscape sides are center-cropped and portrait videos receive black side borders. Full-frame processing is the default; when the person mask is enabled, MediaPipe places the detected person over chroma green before the complete square frame is passed through the selected cartoonization model.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL in a current Chrome or Edge browser for WebGPU acceleration. Drop a video, then select **Run model**. Frames are processed live and never uploaded.

Once a source is ready, selecting a different style automatically loads and runs it. The Run button displays model-loading progress and remains disabled until initialization completes.

The input area also includes a bundled `black.mp4` example and an optional live webcam source. Camera access is requested only after the visitor clicks **Use webcam**, and the stream remains entirely on-device.

Webcam frames are center-cropped or padded and reduced to the selected inference size before being transferred to the inference worker.

Enable **Mirror input** to flip both the source preview and the frames passed into the selected model.

## Live demo

The `main` branch is deployed to [GitHub Pages](https://ldenoue.github.io/animeganv3-browser/) with the workflow in `.github/workflows/deploy-pages.yml`.

## Models

Models are configured in `src/models.ts`. The app includes the White-box Cartoonization `model-33999` ONNX export plus Ghibli C1, Hayao, Shinkai, Comic, and Cute AnimeGANv3 ONNX models in `public/models/`.

`Palette 32` is a model-free custom filter: it lightly blurs and reduces the source to 32×32, extracts a twelve-color palette, smoothly upscales and re-snaps it into flat cel-shaded regions, then clips the result with MediaPipe over the selected background color. Only long, high-confidence internal edges are retained as simplified ink paths; the MediaPipe silhouette is traced separately.

`Cel Shader` is a model-free real-time effect adapted from `cel_shader_demo.html`. It posterizes RGB color and overlays luminance-based Sobel edges, with live controls for color levels, edge sensitivity, and edge thickness. It follows the same crop, mirror, resolution, and optional MediaPipe pipeline as the learned models.

`Contour` is a CPU port of [Filtr's](https://github.com/eurobuddha/filtr) MIT-licensed Contour shader. It defaults to the Filled, 3-level, 0.50-thickness look and also supports the original lines-only mode. See `THIRD_PARTY_NOTICES.md` for attribution.

`Low Poly` builds an edge-guided Delaunay mesh for every frame and fills each triangle with the source color sampled at its centroid. Detail controls mesh density, Edge guide attracts points toward strong luminance gradients, and Irregularity adds deterministic jitter so the facets look organic instead of grid-like.

Turn off **Person mask** to skip MediaPipe processing and stylize the complete framed image. The background color and silhouette outline are then omitted.

AnimeGANv3 and White-box Cartoonization are distributed under their upstream projects' non-commercial licenses. Review the [AnimeGANv3 license](https://github.com/TachibanaYoshino/AnimeGANv3#-license) and [White-box Cartoonization license](https://github.com/SystemErrorWang/White-box-Cartoonization#license) before use.
