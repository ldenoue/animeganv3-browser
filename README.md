# Frame by Frame

Local, in-browser person stylization using MediaPipe multiclass segmentation and AnimeGANv3 through ONNX Runtime Web. WebGPU is selected when available, with a WebAssembly fallback. Frames are mapped to 256×256 while preserving their full height: landscape sides are center-cropped and portrait videos receive black side borders. MediaPipe places the detected person over chroma green before the complete square frame is passed through AnimeGANv3.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL in a current Chrome or Edge browser for WebGPU acceleration. Drop a video, then select **Run AnimeGAN**. Frames are processed live and never uploaded.

## Live demo

The `main` branch is deployed to [GitHub Pages](https://ldenoue.github.io/animeganv3-browser/) with the workflow in `.github/workflows/deploy-pages.yml`.

## Models

Models are configured in `src/models.ts`. The app includes Ghibli C1, Hayao, Shinkai, Comic, and Cute AnimeGANv3 ONNX models in `public/models/`.

`Palette 32` is a model-free custom filter: it lightly blurs and reduces the source to 32×32, extracts a twelve-color palette, smoothly upscales and re-snaps it into flat cel-shaded regions, then clips the result with MediaPipe over the selected background color. Only long, high-confidence internal edges are retained as simplified ink paths; the MediaPipe silhouette is traced separately.

Turn off **Person mask** to skip MediaPipe processing and stylize the complete framed image. The background color and silhouette outline are then omitted.

AnimeGANv3 and its model are distributed under the upstream project's non-commercial license. Review the [AnimeGANv3 license](https://github.com/TachibanaYoshino/AnimeGANv3#-license) before use.
