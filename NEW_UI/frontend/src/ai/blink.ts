import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { AI_CONFIG } from "./config";

let faceLandmarker: FaceLandmarker | null = null;

const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

function dist(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function calcEAR(landmarks: { x: number; y: number }[], indices: number[]): number {
  const p = indices.map((i) => landmarks[i]);
  const v1 = dist(p[1], p[5]);
  const v2 = dist(p[2], p[4]);
  const h = dist(p[0], p[3]);
  return h === 0 ? 0 : (v1 + v2) / (2 * h);
}

function avgVisibility(landmarks: { x: number; y: number; visibility?: number }[], indices: number[]): number {
  let sum = 0;
  for (const i of indices) {
    sum += landmarks[i].visibility ?? 1;
  }
  return sum / indices.length;
}

export async function initBlinkEngine(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: AI_CONFIG.MODELS.FACE_LANDMARKER,
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    numFaces: 10,
    outputFaceBlendshapes: false,
  });
  console.info("[AI] MediaPipe Face Landmarker ready (WASM)");
}

export interface EyeBlinkResult {
  eyesClosed: boolean;
  faceCount: number;
  avgEar: number;
  avgVisibility: number;
}

export async function detectBlink(source: HTMLImageElement | HTMLCanvasElement): Promise<EyeBlinkResult> {
  if (!faceLandmarker) await initBlinkEngine();

  const result = faceLandmarker!.detect(source);
  if (!result.faceLandmarks.length) {
    return { eyesClosed: false, faceCount: 0, avgEar: 1, avgVisibility: 1 };
  }

  let closedCount = 0;
  let totalEar = 0;
  let totalVis = 0;

  for (const face of result.faceLandmarks) {
    const leftEar = calcEAR(face, LEFT_EYE);
    const rightEar = calcEAR(face, RIGHT_EYE);
    const ear = (leftEar + rightEar) / 2;
    const vis = (avgVisibility(face, LEFT_EYE) + avgVisibility(face, RIGHT_EYE)) / 2;

    totalEar += ear;
    totalVis += vis;

    const eyesClosed =
      ear < AI_CONFIG.EAR_THRESHOLD || vis < AI_CONFIG.EYE_VISIBILITY_THRESHOLD;
    if (eyesClosed) closedCount++;
  }

  const faceCount = result.faceLandmarks.length;
  return {
    eyesClosed: closedCount > 0,
    faceCount,
    avgEar: totalEar / faceCount,
    avgVisibility: totalVis / faceCount,
  };
}

export function disposeBlinkEngine(): void {
  faceLandmarker?.close();
  faceLandmarker = null;
}
