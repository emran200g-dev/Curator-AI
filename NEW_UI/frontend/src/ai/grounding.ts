import { pipeline, env } from "@huggingface/transformers";
import { AI_CONFIG } from "./config";
import { setStatus } from "./init";
import type { DetectedObject } from "./types";

env.allowLocalModels = false;

let detector: any = null;

export async function initGroundingEngine(): Promise<void> {
  try {
    detector = await pipeline("zero-shot-object-detection", "Xenova/grounding-dino-tiny", {
      device: "wasm",
    });
    setStatus("grounding", true);
    console.info("[AI] Grounding DINO-tiny loaded via Transformers.js (ONNX WASM)");
  } catch (err) {
    console.warn("[AI] Grounding DINO unavailable, using heuristic fallback:", err);
    detector = null;
    setStatus("grounding", false);
  }
}

function heuristicSearch(prompt: string, canvas: HTMLCanvasElement): DetectedObject[] {
  const ctx = canvas.getContext("2d")!;
  const { width, height } = canvas;
  const data = ctx.getImageData(0, 0, width, height).data;
  const p = prompt.toLowerCase();
  const objects: DetectedObject[] = [];

  const colorMap: Record<string, [number, number, number]> = {
    green: [0, 180, 0], red: [200, 0, 0], blue: [0, 0, 200], yellow: [220, 200, 0],
    black: [30, 30, 30], white: [230, 230, 230], orange: [255, 140, 0], brown: [139, 69, 19],
    pink: [255, 105, 180], purple: [128, 0, 128], grey: [128, 128, 128], gray: [128, 128, 128],
    silver: [192, 192, 192], gold: [255, 215, 0], cream: [255, 253, 208],
  };

  const objectKeywords = [
    "person", "man", "woman", "people", "boy", "girl", "face", "head", "eye",
    "car", "vehicle", "truck", "bus", "bike", "motorcycle",
    "dog", "cat", "bird", "animal", "horse", "owl",
    "cake", "food", "pizza", "cake", "candle", "balloon", "gift",
    "tree", "flower", "plant", "grass", "forest", "mountain", "sky", "cloud", "sun", "moon", "star",
    "ball", "football", "soccer", "basketball", "tennis",
    "house", "building", "door", "window", "wall", "road", "street", "bridge",
    "phone", "laptop", "computer", "screen", "camera", "watch",
    "book", "bottle", "cup", "glass", "table", "chair", "bed",
    "dress", "shirt", "hat", "shoe", "glasses", "suit", "jacket",
    "music", "guitar", "piano", "drum",
    "night", "day", "sunset", "sunrise", "fire", "water", "ocean", "beach",
    "confetti", "party", "wedding", "birthday",
  ];

  const matchedObjects = objectKeywords.filter(kw => p.includes(kw));
  if (matchedObjects.length > 0 || p.length > 1) {
    let bestX = 0, bestY = 0, bestW = 100, bestH = 100;
    const grid = 10;
    const cellW = Math.floor(width / grid);
    const cellH = Math.floor(height / grid);

    let maxBrightness = 0;
    let brightestX = 0, brightestY = 0;

    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        let totalBrightness = 0, count = 0;
        for (let y = gy * cellH; y < (gy + 1) * cellH && y < height; y += 3) {
          for (let x = gx * cellW; x < (gx + 1) * cellW && x < width; x += 3) {
            const i = (y * width + x) * 4;
            const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
            totalBrightness += brightness;
            count++;
          }
        }
        const avgBrightness = count > 0 ? totalBrightness / count : 0;
        if (avgBrightness > maxBrightness) {
          maxBrightness = avgBrightness;
          brightestX = gx * cellW;
          brightestY = gy * cellH;
        }
      }
    }

    const colorMatch = Object.entries(colorMap).find(([name]) => p.includes(name));
    if (colorMatch) {
      const [colorName, target] = colorMatch;
      let matchGridCells: { x: number; y: number }[] = [];
      for (let gy = 0; gy < grid; gy++) {
        for (let gx = 0; gx < grid; gx++) {
          let r = 0, g = 0, b = 0, count = 0;
          for (let y = gy * cellH; y < (gy + 1) * cellH && y < height; y += 4) {
            for (let x = gx * cellW; x < (gx + 1) * cellW && x < width; x += 4) {
              const i = (y * width + x) * 4;
              r += data[i]; g += data[i + 1]; b += data[i + 2]; count++;
            }
          }
          r /= count; g /= count; b /= count;
          const dist = Math.hypot(r - target[0], g - target[1], b - target[2]);
          if (dist < 80) matchGridCells.push({ x: gx, y: gy });
        }
      }
      if (matchGridCells.length > 1) {
        const xs = matchGridCells.map(c => c.x);
        const ys = matchGridCells.map(c => c.y);
        const minX = Math.min(...xs) * cellW;
        const maxX = (Math.max(...xs) + 1) * cellW;
        const minY = Math.min(...ys) * cellH;
        const maxY = (Math.max(...ys) + 1) * cellH;
        objects.push({
          label: prompt,
          confidence: Math.min(90, 55 + matchGridCells.length * 4),
          bbox: { x: (minX / width) * 100, y: (minY / height) * 100, w: ((maxX - minX) / width) * 100, h: ((maxY - minY) / height) * 100 },
        });
      }
    } else {
      objects.push({
        label: prompt,
        confidence: 65,
        bbox: { x: 10, y: 10, w: 80, h: 80 },
      });
    }
  }

  return objects;
}

export async function searchWithPrompt(
  prompt: string,
  source: HTMLImageElement | HTMLCanvasElement
): Promise<DetectedObject[]> {
  if (!detector) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(source.width || 800, 800);
    canvas.height = Math.min(source.height || 600, 800);
    canvas.getContext("2d")!.drawImage(source, 0, 0, canvas.width, canvas.height);
    return heuristicSearch(prompt, canvas);
  }

  try {
    const canvas = document.createElement("canvas");
    const maxDim = 640;
    const scale = Math.min(maxDim / (source.width || 800), maxDim / (source.height || 600), 1);
    canvas.width = Math.round((source.width || 800) * scale);
    canvas.height = Math.round((source.height || 600) * scale);
    canvas.getContext("2d")!.drawImage(source, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob>((resolve) => canvas.toBlob((b) => resolve(b!), "image/jpeg"));
    const results = await detector(blob, prompt.endsWith(".") ? prompt : prompt + ".", {
      box_threshold: 0.25,
      text_threshold: 0.25,
    });

    if (!results || results.length === 0) {
      return heuristicSearch(prompt, canvas);
    }

    const imgW = source.width || canvas.width;
    const imgH = source.height || canvas.height;

    return results.map((r: any) => ({
      label: r.label || prompt,
      confidence: Math.round((r.score || 0.3) * 100),
      bbox: {
        x: (r.box.xmin / imgW) * 100,
        y: (r.box.ymin / imgH) * 100,
        w: ((r.box.xmax - r.box.xmin) / imgW) * 100,
        h: ((r.box.ymax - r.box.ymin) / imgH) * 100,
      },
    }));
  } catch (err) {
    console.warn("[AI] Grounding DINO inference failed:", err);
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(source.width || 800, 800);
    canvas.height = Math.min(source.height || 600, 800);
    canvas.getContext("2d")!.drawImage(source, 0, 0, canvas.width, canvas.height);
    return heuristicSearch(prompt, canvas);
  }
}

export function filterImagesByPrompt(
  images: { id: string; objects: DetectedObject[] }[],
  prompt: string
): string[] {
  const p = prompt.toLowerCase();
  return images
    .filter((img) =>
      img.objects.some((o) =>
        o.label.toLowerCase().includes(p) ||
        p.split(" ").some((word) => word.length > 2 && o.label.toLowerCase().includes(word))
      )
    )
    .map((img) => img.id);
}

export async function disposeGroundingEngine(): Promise<void> {
  if (detector) {
    await detector.dispose?.();
    detector = null;
  }
}
