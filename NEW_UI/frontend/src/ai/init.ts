import { AI_CONFIG } from "./config";
import { initBlurEngine } from "./blur";
import { initBlinkEngine } from "./blink";
import type { AIEngineStatus } from "./types";

let status: AIEngineStatus = {
  blur: false,
  blink: false,
  burst: false,
  grounding: false,
};

export function getAIStatus(): AIEngineStatus {
  return { ...status };
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms)
    ),
  ]);
}

/** Init only blur + blink (fast). Burst and grounding init lazily on first use. */
export async function initAIEngines(): Promise<boolean> {
  const results = await Promise.allSettled([
    withTimeout(initBlurEngine(), 30000, "Blur engine").then(() => { status.blur = true; }),
    withTimeout(initBlinkEngine(), 60000, "Blink engine").then(() => { status.blink = true; }),
  ]);

  const failed = results.filter((r) => r.status === "rejected");
  if (failed.length) {
    console.warn("[AI] Critical engine init failed:", failed.map((r) => r.status === "rejected" ? r.reason?.message : "unknown"));
  }

  return status.blur && status.blink;
}

export function setStatus(key: keyof AIEngineStatus, val: boolean): void {
  status[key] = val;
}
