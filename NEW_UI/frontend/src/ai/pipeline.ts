import { scoreSharpness, isBlurry } from "./blur";
import { detectBlink } from "./blink";
import { embedImage, groupBursts } from "./burst";
import { initAIEngines } from "./init";
import type { ImageAnalysisResult, SessionResults } from "./types";

export interface PipelineImage {
  id: string;
  file: File;
  name: string;
  url: string;
}

export interface PipelineProgress {
  phase: string;
  current: string;
  percent: number;
}

function loadImageElement(src: string | File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = window.document.createElement("img");
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${typeof src === "string" ? src : src.name}`));
    img.src = typeof src === "string" ? src : URL.createObjectURL(src);
    setTimeout(() => reject(new Error("Image load timeout")), 15000);
  });
}

function categorize(
  sharpness: number,
  eyesClosed: boolean,
  isDuplicate: boolean
): ImageAnalysisResult["category"] {
  if (isDuplicate) return "duplicate";
  if (isBlurry(sharpness)) return "blurry";
  if (eyesClosed) return "rejected";
  return "keeper";
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms)
    ),
  ]);
}

export async function runAnalysisPipeline(
  images: PipelineImage[],
  title: string,
  eventDate: string,
  onProgress: (p: PipelineProgress) => void
): Promise<SessionResults> {
  await withTimeout(initAIEngines(), 120000, "AI init");

  const sharpnessScores: number[] = [];
  const embeddings: Float32Array[] = [];
  const blinkResults: Awaited<ReturnType<typeof detectBlink>>[] = [];
  const elements: HTMLImageElement[] = [];
  const failedImages = new Set<string>();

  // Phase 1: Blur + blink per image (skip failures)
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    onProgress({
      phase: "Blur & blink detection",
      current: img.name,
      percent: Math.round((i / images.length) * 50),
    });

    try {
      const el = await withTimeout(loadImageElement(img.file), 15000, `Load ${img.name}`);
      elements.push(el);

      const [sharpness, blink] = await Promise.all([
        withTimeout(scoreSharpness(el), 10000, `Sharpness ${img.name}`),
        withTimeout(detectBlink(el), 10000, `Blink ${img.name}`),
      ]);

      sharpnessScores.push(sharpness);
      blinkResults.push(blink);
    } catch (err) {
      console.warn(`[Pipeline] Skipping ${img.name}:`, err);
      failedImages.add(img.id);
      sharpnessScores.push(0.5);
      blinkResults.push({ eyesClosed: false, faceCount: 0, avgEar: 0.3, avgVisibility: 1 });
    }
  }

  // Phase 2: DINOv2 embeddings + burst grouping (skip failures)
  onProgress({ phase: "DINOv2 embedding", current: "Generating embeddings...", percent: 55 });
  for (let i = 0; i < elements.length; i++) {
    onProgress({
      phase: "DINOv2 embedding",
      current: images[i].name,
      percent: 55 + Math.round((i / elements.length) * 25),
    });

    if (failedImages.has(images[i].id)) {
      embeddings.push(new Float32Array(768));
      continue;
    }

    try {
      embeddings.push(await withTimeout(embedImage(elements[i]), 20000, `Embed ${images[i].name}`));
    } catch (err) {
      console.warn(`[Pipeline] Embed failed for ${images[i].name}:`, err);
      embeddings.push(new Float32Array(768));
    }
  }

  const burstGroups = groupBursts(embeddings, sharpnessScores);
  const duplicateSet = new Set<number>();
  const bestInGroup = new Map<number, number>();

  for (const g of burstGroups) {
    bestInGroup.set(g.groupId, g.bestIndex);
    for (const idx of g.indices) {
      if (idx !== g.bestIndex) duplicateSet.add(idx);
    }
  }

  // Phase 3: Assign categories
  onProgress({ phase: "Categorizing", current: "Finalizing...", percent: 85 });

  const results: ImageAnalysisResult[] = images.map((img, i) => {
    const sharpness = sharpnessScores[i];
    const blink = blinkResults[i];
    const isDuplicate = duplicateSet.has(i);
    const failed = failedImages.has(img.id);
    const category = failed ? "keeper" : categorize(sharpness, blink.eyesClosed, isDuplicate);

    let caption = "";
    if (failed) caption = "Analysis skipped (load timeout).";
    else if (category === "keeper") caption = `Sharp keeper — score ${sharpness.toFixed(2)}, eyes open.`;
    else if (category === "duplicate") caption = `Burst duplicate — grouped with ${images[bestInGroup.get(burstGroups.find(g => g.indices.includes(i))?.groupId ?? 0) ?? i]?.name}.`;
    else if (category === "blurry") caption = `Blurry — sharpness ${sharpness.toFixed(2)} below 0.30 threshold.`;
    else caption = `Rejected — eyes closed (EAR ${blink.avgEar.toFixed(2)}, visibility ${blink.avgVisibility.toFixed(2)}).`;

    const group = burstGroups.find((g) => g.indices.includes(i));

    return {
      id: img.id,
      filename: img.name,
      category,
      sharpnessScore: sharpness,
      confidence: Math.round(sharpness * 100),
      caption,
      eyesClosed: blink.eyesClosed,
      isDuplicate,
      embedding: Array.from(embeddings[i].slice(0, 32)),
      objects: [],
      objectsLoaded: false,
      burstGroupId: group?.groupId,
    };
  });

  onProgress({ phase: "Complete", current: "", percent: 100 });

  return {
    session_id: "",
    title,
    eventDate,
    analyzedAt: new Date().toISOString(),
    images: results,
  };
}
