# Frame by Frame

Local, in-browser video stylization using a framework-free HTML and TypeScript interface, optional MediaPipe multiclass segmentation, and ONNX Runtime Web. WebGPU is selected when available, with a WebAssembly fallback. Frames are mapped to 256×256 while preserving their full height: landscape sides are center-cropped and portrait videos receive black side borders. Full-frame processing is the default; when the person mask is enabled, MediaPipe places the detected person over chroma green before the complete square frame is passed through the selected cartoonization model.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL in a current Chrome or Edge browser for WebGPU acceleration. Drop a video, then select **Run model**. Frames are processed live and never uploaded.

The upload area also includes a bundled `black.mp4` example so visitors can try the pipeline immediately without choosing a local file.

## Live demo

The `main` branch is deployed to [GitHub Pages](https://ldenoue.github.io/animeganv3-browser/) with the workflow in `.github/workflows/deploy-pages.yml`.

## Models

Models are configured in `src/models.ts`. The app includes the White-box Cartoonization `model-33999` ONNX export plus Ghibli C1, Hayao, Shinkai, Comic, and Cute AnimeGANv3 ONNX models in `public/models/`.

`Palette 32` is a model-free custom filter: it lightly blurs and reduces the source to 32×32, extracts a twelve-color palette, smoothly upscales and re-snaps it into flat cel-shaded regions, then clips the result with MediaPipe over the selected background color. Only long, high-confidence internal edges are retained as simplified ink paths; the MediaPipe silhouette is traced separately.

Turn off **Person mask** to skip MediaPipe processing and stylize the complete framed image. The background color and silhouette outline are then omitted.

AnimeGANv3 and White-box Cartoonization are distributed under their upstream projects' non-commercial licenses. Review the [AnimeGANv3 license](https://github.com/TachibanaYoshino/AnimeGANv3#-license) and [White-box Cartoonization license](https://github.com/SystemErrorWang/White-box-Cartoonization#license) before use.
