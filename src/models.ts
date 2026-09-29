export type AnimeModel = {
  id: string;
  label: string;
  description: string;
  url?: string;
  engine?: "animegan" | "palette";
  stride?: 8 | 16;
};

export const MODELS: AnimeModel[] = [
  {
    id: "hayao-36",
    label: "Hayao",
    description: "Warm, painterly landscape style",
    url: "/models/AnimeGANv3_Hayao_36.onnx",
  },
  {
    id: "ghibli-c1",
    label: "Ghibli C1",
    description: "Warm, hand-painted Ghibli portrait style",
    url: "/models/AnimeGANv3_large_Ghibli_c1_e299.onnx",
  },
  {
    id: "shinkai-37",
    label: "Shinkai",
    description: "Crisp light and cinematic color",
    url: "/models/AnimeGANv3_Shinkai_37.onnx",
  },
  {
    id: "comic-jp-face",
    label: "Comic",
    description: "Japanese comic portrait style",
    url: "/models/AnimeGANv3_JP_face_v1.0.onnx",
  },
  {
    id: "cute-tiny",
    label: "Cute",
    description: "Soft, playful cartoon portraits",
    url: "/models/AnimeGANv3_tiny_Cute.onnx",
    stride: 16,
  },
  {
    id: "palette-32",
    label: "Palette 32",
    description: "Flat cel palette with selective ink",
    engine: "palette",
  },
];
