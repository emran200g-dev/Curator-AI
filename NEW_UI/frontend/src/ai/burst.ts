import { pipeline, env } from "@huggingface/transformers";
import { AI_CONFIG } from "./config";
import { setStatus } from "./init";

env.allowLocalModels = false;

let featureExtractor: any = null;
let useTfFallback = false;
let initPromise: Promise<void> | null = null;

async function doInit(): Promise<void> {
  try {
    featureExtractor = await pipeline("image-feature-extraction", "Xenova/dinov2-small", {
      device: "wasm",
    });
    setStatus("burst", true);
    console.info("[AI] DINOv2-small loaded via Transformers.js (ONNX WASM)");
  } catch (err) {
    console.warn("[AI] DINOv2 unavailable, using TF.js embedding fallback:", err);
    useTfFallback = true;
    setStatus("burst", true);
  }
}

export async function initBurstEngine(): Promise<void> {
  if (!initPromise) initPromise = doInit();
  return initPromise;
}

async function embedWithDino(source: HTMLImageElement | HTMLCanvasElement): Promise<Float32Array> {
  const canvas = window.document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  canvas.getContext("2d")!.drawImage(source, 0, 0, 224, 224);
  const blob = await new Promise<Blob>((resolve) => canvas.toBlob((b) => resolve(b!), "image/jpeg"));
  const result = await featureExtractor(blob);
  return new Float32Array(result.data);
}

async function embedWithTfFallback(source: HTMLImageElement | HTMLCanvasElement): Promise<Float32Array> {
  const tf = await import("@tensorflow/tfjs") as any;
  return tf.tidy(() => {
    const t = tf.browser.fromPixels(source).resizeBilinear([224, 224]).toFloat().div(255);
    const g = t.mean(2, true);
    const g4d = g.expandDims(0);
    const pool1 = tf.avgPool(g4d, 4, 4, "valid");
    const pool2 = tf.maxPool(g4d, 8, 8, "valid");
    const lapKernel = tf.tensor4d(new Float32Array([0, 1, 0, 1, -4, 1, 0, 1, 0]), [3, 3, 1, 1]);
    const edges = tf.conv2d(g4d, lapKernel, 1, "same");
    const combined = tf.concat([
      tf.reshape(pool1, [1, -1]),
      tf.reshape(pool2, [1, -1]),
      tf.reshape(edges, [1, -1]),
    ], 1);
    const normalized = tf.div(combined, tf.norm(combined, 2, 1, true).add(1e-6));
    return new Float32Array(normalized.dataSync());
  });
}

export async function embedImage(source: HTMLImageElement | HTMLCanvasElement): Promise<Float32Array> {
  if (!featureExtractor && !useTfFallback) await initBurstEngine();
  if (featureExtractor && !useTfFallback) {
    try {
      return await embedWithDino(source);
    } catch {
      useTfFallback = true;
    }
  }
  return embedWithTfFallback(source);
}

export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

export interface BurstGroup {
  groupId: number;
  indices: number[];
  bestIndex: number;
}

export function groupBursts(
  embeddings: Float32Array[],
  sharpnessScores: number[],
  threshold = AI_CONFIG.BURST_SIMILARITY
): BurstGroup[] {
  const n = embeddings.length;
  const visited = new Array(n).fill(false);
  const groups: BurstGroup[] = [];
  let groupId = 0;

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    const indices = [i];
    visited[i] = true;

    for (let j = i + 1; j < n; j++) {
      if (visited[j]) continue;
      const sim = cosineSimilarity(embeddings[i], embeddings[j]);
      if (sim >= threshold) {
        indices.push(j);
        visited[j] = true;
      }
    }

    const bestIndex = indices.reduce((best, idx) =>
      sharpnessScores[idx] > sharpnessScores[best] ? idx : best, indices[0]);

    groups.push({ groupId: groupId++, indices, bestIndex });
  }

  return groups;
}

export async function disposeBurstEngine(): Promise<void> {
  if (featureExtractor) {
    await featureExtractor.dispose?.();
    featureExtractor = null;
  }
}
