import * as tf from "@tensorflow/tfjs";
import { AI_CONFIG } from "./config";

let iqaModel: tf.LayersModel | null = null;
let hasTrainedModel = false;

function buildIQARegressor(): tf.LayersModel {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [224, 224, 3], filters: 16, kernelSize: 3, activation: "relu", padding: "same" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu", padding: "same" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu", padding: "same" }));
  model.add(tf.layers.globalAveragePooling2d({}));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  return model;
}

export async function initBlurEngine(): Promise<void> {
  await tf.ready();
  try {
    iqaModel = await tf.loadLayersModel(AI_CONFIG.MODELS.IQA);
    hasTrainedModel = true;
    console.info("[AI] Loaded trained IQA model from", AI_CONFIG.MODELS.IQA);
  } catch {
    iqaModel = buildIQARegressor();
    hasTrainedModel = false;
    console.info("[AI] No trained IQA model — using multi-metric sharpness");
  }
}

function laplacianVariance(source: HTMLImageElement | HTMLCanvasElement): number {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(source).resizeBilinear([224, 224]).toFloat().div(255);
    const gray = img.mean(2);
    const kernel = tf.tensor4d([-1, -1, -1, -1, 8, -1, -1, -1, -1], [3, 3, 1, 1]);
    const gray4d = gray.expandDims(0).expandDims(3) as tf.Tensor4D;
    const lap = tf.conv2d(gray4d, kernel, 1, "same");
    const absLap = lap.abs();
    const meanVal = absLap.mean();
    const variance = absLap.sub(meanVal).square().mean();
    return variance.dataSync()[0];
  });
}

function edgeDensity(source: HTMLImageElement | HTMLCanvasElement): number {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(source).resizeBilinear([224, 224]).toFloat().div(255);
    const gray = img.mean(2);
    const gxData = new Float32Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]);
    const gyData = new Float32Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]);
    const gx = tf.tensor4d(gxData, [3, 3, 1, 1]);
    const gy = tf.tensor4d(gyData, [3, 3, 1, 1]);
    const gray4d = gray.expandDims(0).expandDims(3) as tf.Tensor4D;
    const gxAbs = tf.conv2d(gray4d, gx, 1, "same").abs();
    const gyAbs = tf.conv2d(gray4d, gy, 1, "same").abs();
    const mag = gxAbs.square().add(gyAbs.square()).sqrt();
    const edgePixels = mag.greater(0.15).sum().dataSync()[0];
    return edgePixels / (224 * 224);
  });
}

function localContrast(source: HTMLImageElement | HTMLCanvasElement): number {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(source).resizeBilinear([224, 224]).toFloat().div(255);
    const gray = img.mean(2);
    const blocks = 8;
    const bh = Math.floor(224 / blocks);
    const bw = Math.floor(224 / blocks);
    let totalVar = 0;
    let count = 0;
    for (let by = 0; by < blocks; by++) {
      for (let bx = 0; bx < blocks; bx++) {
        const block = gray.slice([by * bh, bx * bw], [bh, bw]);
        const blockVar = block.sub(block.mean()).square().mean().dataSync()[0];
        totalVar += blockVar;
        count++;
      }
    }
    return Math.sqrt(totalVar / count);
  });
}

export async function scoreSharpness(source: HTMLImageElement | HTMLCanvasElement): Promise<number> {
  if (!iqaModel) await initBlurEngine();

  console.log("[AI] Blur scoring: trainedModel =", hasTrainedModel);
  if (hasTrainedModel) {
    const cnnScore = tf.tidy(() => {
      const input = tf.browser.fromPixels(source).resizeBilinear([224, 224]).toFloat().div(255).expandDims(0) as tf.Tensor4D;
      const pred = iqaModel!.predict(input) as tf.Tensor;
      return pred.dataSync()[0];
    });
    const lapVar = laplacianVariance(source);
    const lapScore = Math.min(1, lapVar * 8);
    return Math.min(1, Math.max(0, cnnScore * 0.4 + lapScore * 0.6));
  }

  const lapVar = laplacianVariance(source);
  const edge = edgeDensity(source);
  const contrast = localContrast(source);
  const lapNorm = Math.min(1, lapVar * 5);
  const edgeNorm = Math.min(1, edge * 2);
  const contrastNorm = Math.min(1, contrast * 2);
  const combined = lapNorm * 0.55 + edgeNorm * 0.3 + contrastNorm * 0.15;
  const score = Math.min(1, Math.max(0, combined));
  console.log("[AI] Sharpness: lapVar=", lapVar.toFixed(4), "edge=", edge.toFixed(4), "contrast=", contrast.toFixed(4), "=> score=", score.toFixed(3));
  return score;
}

export function isBlurry(score: number): boolean {
  return score < AI_CONFIG.BLUR_THRESHOLD;
}

export function disposeBlurEngine(): void {
  iqaModel?.dispose();
  iqaModel = null;
}
