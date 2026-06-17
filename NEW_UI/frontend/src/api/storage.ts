import { AI_CONFIG } from "../ai/config";
import type { SessionResults } from "../ai/types";
import JSZip from "jszip";

const BASE = AI_CONFIG.STORAGE_API;

async function req(path: string, options: RequestInit = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
  return res;
}

export async function createSession(): Promise<string> {
  const res = await req("/sessions", { method: "POST" });
  const data = await res.json();
  return data.session_id;
}

export async function uploadFiles(sessionId: string, files: File[]): Promise<string[]> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await req(`/sessions/${sessionId}/upload`, { method: "POST", body: form });
  const data = await res.json();
  return data.uploaded;
}

export function fileUrl(sessionId: string, filename: string): string {
  return `${BASE}/sessions/${sessionId}/files/${encodeURIComponent(filename)}`;
}

export async function saveResults(sessionId: string, results: SessionResults): Promise<void> {
  await req(`/sessions/${sessionId}/results.json`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...results, session_id: sessionId }),
  });
}

export async function detectObjects(file: File): Promise<{ label: string; confidence: number; bbox: { x: number; y: number; w: number; h: number } }[]> {
  const form = new FormData();
  form.append("file", file);
  const res = await req("/api/detect", { method: "POST", body: form });
  const data = await res.json();
  return data.objects || [];
}

export async function captionImage(file: File): Promise<string[]> {
  const form = new FormData();
  form.append("file", file);
  const res = await req("/api/caption", { method: "POST", body: form });
  const data = await res.json();
  return data.captions || [];
}

export async function classifyImage(file: File): Promise<{ label: string; score: number }[]> {
  const form = new FormData();
  form.append("file", file);
  const res = await req("/api/classify", { method: "POST", body: form });
  const data = await res.json();
  return data.labels || [];
}

export async function downloadSessionZip(
  sessionId: string,
  title: string,
  keepers: { filename: string; url: string }[]
): Promise<void> {
  const zip = new JSZip();
  for (const k of keepers) {
    const res = await fetch(k.url);
    if (res.ok) zip.file(k.filename, await res.blob());
  }
  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `Curated_${title.replace(/\s+/g, "_")}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}
