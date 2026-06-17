export type PhotoCategory = "keeper" | "duplicate" | "blurry" | "rejected";

export interface DetectedObject {
  label: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
}

export interface ImageAnalysisResult {
  id: string;
  filename: string;
  category: PhotoCategory;
  sharpnessScore: number;
  confidence: number;
  caption: string;
  eyesClosed: boolean;
  isDuplicate: boolean;
  embedding?: number[];
  objects: DetectedObject[];
  objectsLoaded: boolean;
  burstGroupId?: number;
}

export interface SessionResults {
  session_id: string;
  title: string;
  eventDate: string;
  analyzedAt: string;
  images: ImageAnalysisResult[];
}

export interface AIEngineStatus {
  blur: boolean;
  blink: boolean;
  burst: boolean;
  grounding: boolean;
}
