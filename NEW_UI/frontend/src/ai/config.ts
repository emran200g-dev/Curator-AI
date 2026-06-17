/** Central config */
export const AI_CONFIG = {
  /** Sharpness threshold — trash below this (0-1 scale) */
  BLUR_THRESHOLD: 0.3,

  /** Burst grouping cosine similarity threshold */
  BURST_SIMILARITY: 0.9,

  /** MediaPipe eye visibility threshold for closed-eye detection */
  EYE_VISIBILITY_THRESHOLD: 0.65,

  /** EAR threshold for blink detection */
  EAR_THRESHOLD: 0.21,

  /** Model paths (place ONNX files in public/models/ or set CDN URLs) */
  MODELS: {
    IQA: "/models/iqa/model.json",
    DINOV2_ONNX: import.meta.env.VITE_DINOV2_MODEL_URL || "/models/dinov2-small.onnx",
    GROUNDING_DINO_ONNX: import.meta.env.VITE_GROUNDING_MODEL_URL || "/models/grounding-dino-tiny.onnx",
    FACE_LANDMARKER: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  },

  STORAGE_API: import.meta.env.VITE_STORAGE_API_URL || "http://localhost:7860",
};
