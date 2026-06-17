import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  FolderOpen,
  Home,
  LayoutDashboard,
  Image,
  Camera,
  Eye,
  EyeOff,
  X,
  Archive,
  Sparkles,
  Calendar,
  Search,
  CheckCircle,
  Copy,
  Trash2,
  Zap,
  List,
  Target,
  Tag,
  ChevronRight,
  LayoutGrid,
  Filter,
  Play,
  Scan,
  MessageSquare,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { toast, Toaster } from "sonner";
import { runAnalysisPipeline } from "../ai/pipeline";
import { initAIEngines } from "../ai/init";
import { searchWithPrompt, filterImagesByPrompt } from "../ai/grounding";
import { AI_CONFIG } from "../ai/config";
import type { DetectedObject, PhotoCategory } from "../ai/types";
import {
  createSession,
  uploadFiles,
  saveResults,
  downloadSessionZip,
  fileUrl,
  detectObjects,
  captionImage,
  classifyImage,
} from "../api/storage";

// ─── Types ────────────────────────────────────────────────────────────────────

type Category = PhotoCategory;
type Screen = "home" | "app";
type Tab = "dashboard" | "gallery" | "objects" | "caption";
type GalleryFilter = "all" | Category;

interface UploadedImage {
  id: string;
  file: File;
  url: string;
  name: string;
  size: number;
  photoId?: number; // unused — kept for compat
}

interface AnalyzedImage extends UploadedImage {
  category: Category;
  caption: string;
  confidence: number;
  objects: DetectedObject[];
  objectsLoaded: boolean;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const CATEGORY_CONFIG = {
  keeper: {
    label: "Keeper",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
    badgeBg: "bg-emerald-500/20 text-emerald-300",
    icon: CheckCircle,
    hex: "#22c55e",
  },
  duplicate: {
    label: "Duplicate",
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
    badgeBg: "bg-blue-500/20 text-blue-300",
    icon: Copy,
    hex: "#3b82f6",
  },
  blurry: {
    label: "Blurry / Bad",
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
    badgeBg: "bg-amber-500/20 text-amber-300",
    icon: EyeOff,
    hex: "#f59e0b",
  },
  rejected: {
    label: "Rejected",
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    border: "border-purple-500/20",
    badgeBg: "bg-purple-500/20 text-purple-300",
    icon: Trash2,
    hex: "#a855f7",
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

const folderInputProps = { webkitdirectory: "", directory: "" } as React.InputHTMLAttributes<HTMLInputElement>;

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");

  const [projectTitle, setProjectTitle] = useState("");
  const [eventDate, setEventDate] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [cleanupCount, setCleanupCount] = useState<number | null>(null);
  const [aiReady, setAiReady] = useState(false);
  const [nlPrompt, setNlPrompt] = useState("");
  const [objectFilterIds, setObjectFilterIds] = useState<string[] | null>(null);

  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const [analysisState, setAnalysisState] = useState<"idle" | "analyzing" | "complete">("idle");
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analyzedImages, setAnalyzedImages] = useState<AnalyzedImage[]>([]);
  const [currentFile, setCurrentFile] = useState("");

  const [galleryFilter, setGalleryFilter] = useState<GalleryFilter>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [captions, setCaptions] = useState<Record<string, string>>({});
  const [editingCaption, setEditingCaption] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const [selectedImage, setSelectedImage] = useState<AnalyzedImage | null>(null);
  const [showBboxes, setShowBboxes] = useState(true);
  const [objectSearch, setObjectSearch] = useState("");
  const [objectDetectingId, setObjectDetectingId] = useState<string | null>(null);
  const [captionGenerating, setCaptionGenerating] = useState(false);
  const [generatedCaptions, setGeneratedCaptions] = useState<Record<string, string[]>>({});
  const [detectingImageId, setDetectingImageId] = useState<string | null>(null);
  const [refImageFile, setRefImageFile] = useState<File | null>(null);
  const [refImageSimilarity, setRefImageSimilarity] = useState<Record<string, number> | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    initAIEngines().then(() => setAiReady(true));
  }, []);

  // ── File handlers ─────────────────────────────────────────────────────────

  const processFiles = useCallback((files: File[], uploadMethod = "unknown") => {
    const images = files.filter((f) => f.type.startsWith("image/"));
    if (!images.length) { toast.error("No image files found"); return; }
    const fileTypes = new Set<string>();
    images.forEach(f => { const ext = f.name.split('.').pop()?.toLowerCase(); if (ext) fileTypes.add(ext); });
    let addedCount = 0;
    let addedSizeBytes = 0;
    let skippedCount = 0;
    setUploadedImages((prev) => {
      const existing = new Set(prev.map((p) => p.name + p.size));
      const fresh: UploadedImage[] = images
        .filter((f) => !existing.has(f.name + f.size))
        .map((f, i) => ({
          id: `${Date.now()}-${i}-${Math.random()}`,
          file: f,
          url: URL.createObjectURL(f),
          name: f.name,
          size: f.size,
        }));
      addedCount = fresh.length;
      addedSizeBytes = fresh.reduce((sum, f) => sum + f.size, 0);
      skippedCount = images.length - fresh.length;
      if (fresh.length) toast.success(`Added ${fresh.length} image${fresh.length !== 1 ? "s" : ""}`);
      else toast.info("Images already in list");
      return [...prev, ...fresh];
    });
    if (addedCount > 0) {
      pendo.track("images_uploaded", {
        imageCount: addedCount,
        totalSizeBytes: addedSizeBytes,
        uploadMethod,
        duplicatesSkipped: skippedCount,
        fileTypes: Array.from(fileTypes).join(","),
      });
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); processFiles(Array.from(e.dataTransfer.files), "drag_drop"); },
    [processFiles]
  );

  const startAnalysis = useCallback(async () => {
    if (!uploadedImages.length) { toast.error("Upload at least one image first"); return; }
    if (!projectTitle.trim()) { toast.error("Enter an album title first"); return; }

    setAnalysisState("analyzing");
    setAnalysisProgress(0);

    pendo.track("project_analysis_started", {
      projectTitle: projectTitle.trim().substring(0, 100),
      eventDate,
      imageCount: uploadedImages.length,
      sessionId: sessionId || "",
      totalUploadSizeBytes: uploadedImages.reduce((sum, img) => sum + img.size, 0),
    });

    try {
      let sid = sessionId;
      if (!sid) {
        sid = await createSession();
        setSessionId(sid);
        setCurrentFile("Uploading to storage...");
        await uploadFiles(sid, uploadedImages.map((i) => i.file));
      }

      const pipelineImages = uploadedImages.map((img) => ({
        id: img.id,
        file: img.file,
        name: img.name,
        url: img.url,
      }));

      const results = await runAnalysisPipeline(
        pipelineImages,
        projectTitle.trim(),
        eventDate,
        ({ phase, current, percent }) => {
          setCurrentFile(current ? `${phase}: ${current}` : phase);
          setAnalysisProgress(percent);
        }
      );

      const mapped: AnalyzedImage[] = results.images.map((r) => {
        const existing = uploadedImages.find((u) => u.name === r.filename || u.id === r.id);
        const url = existing?.url ?? (sid ? fileUrl(sid, r.filename) : "");
        return {
          id: r.id,
          file: existing?.file ?? new File([], r.filename),
          url,
          name: r.filename,
          size: existing?.size ?? 0,
          category: r.category,
          caption: r.caption,
          confidence: r.confidence,
          objects: r.objects,
          objectsLoaded: r.objectsLoaded,
        };
      });

      if (sid) {
        await saveResults(sid, { ...results, session_id: sid });
      }

      setAnalyzedImages(mapped);
      setAnalysisState("complete");
      setAnalysisProgress(100);
      setSelectedImage(mapped.find((r) => r.category === "keeper") ?? mapped[0] ?? null);

      const k = mapped.filter((r) => r.category === "keeper").length;
      toast.success(`Analysis complete — ${k} keeper${k !== 1 ? "s" : ""} found`);

      pendo.track("analysis_pipeline_completed", {
        projectTitle: projectTitle.trim().substring(0, 100),
        eventDate,
        sessionId: sid || "",
        totalImages: mapped.length,
        keeperCount: k,
        duplicateCount: mapped.filter((r) => r.category === "duplicate").length,
        blurryCount: mapped.filter((r) => r.category === "blurry").length,
        rejectedCount: mapped.filter((r) => r.category === "rejected").length,
      });

      setActiveTab("gallery");
    } catch (err) {
      console.error(err);
      setAnalysisState("idle");
      toast.error(err instanceof Error ? err.message : "Analysis failed");
      pendo.track("analysis_pipeline_failed", {
        errorMessage: (err instanceof Error ? err.message : "Analysis failed").substring(0, 200),
        projectTitle: projectTitle.trim().substring(0, 100),
        imageCount: uploadedImages.length,
        sessionId: sessionId || "",
      });
    }
  }, [uploadedImages, projectTitle, eventDate, sessionId]);

  const runObjectDetection = useCallback(async (img: AnalyzedImage) => {
    if (!nlPrompt.trim()) {
      toast.error("Enter a search prompt first (e.g. 'person with green hat')");
      return;
    }
    if (objectDetectingId === img.id) return;

    setObjectDetectingId(img.id);
    try {
      const el = await new Promise<HTMLImageElement>((resolve, reject) => {
        const im = window.document.createElement("img");
        im.crossOrigin = "anonymous";
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = img.url;
      });

      const objects = await searchWithPrompt(nlPrompt.trim(), el);
      const updated: AnalyzedImage = { ...img, objects, objectsLoaded: true };

      setAnalyzedImages((prev) => prev.map((p) => (p.id === img.id ? updated : p)));
      setSelectedImage((prev) => (prev?.id === img.id ? updated : prev));

      toast.success(objects.length ? `Found ${objects.length} match(es) for "${nlPrompt}"` : "No matches found");
      pendo.track("nl_object_search_completed", {
        searchPrompt: nlPrompt.trim().substring(0, 100),
        resultsCount: objects.length,
        imageName: img.name,
        imageId: img.id,
        matchFound: objects.length > 0,
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Object search failed");
    } finally {
      setObjectDetectingId(null);
    }
  }, [nlPrompt, objectDetectingId]);

  const runGlobalObjectSearch = useCallback(async () => {
    if (!nlPrompt.trim()) { toast.error("Enter a search prompt"); return; }
    setObjectDetectingId("all");
    try {
      const updated = await Promise.all(
        analyzedImages.map(async (img) => {
      const el = await new Promise<HTMLImageElement>((resolve, reject) => {
        const im = window.document.createElement("img");
        im.crossOrigin = "anonymous";
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = img.url;
          });
          const objects = await searchWithPrompt(nlPrompt.trim(), el);
          return { ...img, objects, objectsLoaded: true };
        })
      );
      setAnalyzedImages(updated);
      const ids = filterImagesByPrompt(updated, nlPrompt);
      setObjectFilterIds(ids);
      toast.success(`Found ${ids.length} image(s) matching "${nlPrompt}"`);
      pendo.track("global_object_search_completed", {
        searchPrompt: nlPrompt.trim().substring(0, 100),
        totalImages: analyzedImages.length,
        matchingImagesCount: ids.length,
        totalObjectsFound: updated.reduce((sum, img) => sum + img.objects.length, 0),
      });
      setActiveTab("objects");
    } catch {
      toast.error("Search failed");
    } finally {
      setObjectDetectingId(null);
    }
  }, [analyzedImages, nlPrompt]);

  const runDetectSingle = useCallback(async (img: AnalyzedImage) => {
    if (detectingImageId) return;
    setDetectingImageId(img.id);
    try {
      const el = await new Promise<HTMLImageElement>((resolve, reject) => {
        const im = window.document.createElement("img");
        im.crossOrigin = "anonymous";
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = img.url;
      });
      const objects = await detectObjects(img.file);
      const mapped: DetectedObject[] = objects.map((o) => ({
        label: o.label,
        confidence: o.confidence,
        bbox: {
          x: (o.bbox.x / el.naturalWidth) * 100,
          y: (o.bbox.y / el.naturalHeight) * 100,
          w: (o.bbox.w / el.naturalWidth) * 100,
          h: (o.bbox.h / el.naturalHeight) * 100,
        },
      }));
      const updated: AnalyzedImage = { ...img, objects: mapped, objectsLoaded: true };
      setAnalyzedImages((prev) => prev.map((p) => p.id === img.id ? updated : p));
      setSelectedImage((prev) => prev?.id === img.id ? updated : prev);
      toast.success(objects.length ? `Detected ${objects.length} object(s)` : "No objects detected");
      pendo.track("object_detection_completed", {
        imageName: img.name,
        imageId: img.id,
        objectsDetectedCount: objects.length,
        objectLabels: objects.map((o) => o.label).join(",").substring(0, 200),
        imageCategory: img.category,
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setDetectingImageId(null);
    }
  }, [detectingImageId]);

  const handleSelectImage = useCallback((img: AnalyzedImage) => {
    setSelectedImage(img);
  }, []);

  const runCaptionSingle = useCallback(async (img: AnalyzedImage) => {
    setCaptionGenerating(true);
    try {
      const captions = await captionImage(img.file);
      setGeneratedCaptions((prev) => ({ ...prev, [img.id]: captions }));
      if (captions.length > 0) {
        setCaptions((prev) => ({ ...prev, [img.id]: captions[0] }));
      }
      toast.success(captions.length ? `${captions.length} captions generated` : "No captions returned");
      pendo.track("caption_generated", {
        imageName: img.name,
        imageId: img.id,
        captionCount: captions.length,
        imageCategory: img.category,
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Caption failed");
    } finally {
      setCaptionGenerating(false);
    }
  }, []);

  const runCaptionAll = useCallback(async () => {
    setCaptionGenerating(true);
    let done = 0;
    try {
      for (const img of analyzedImages) {
        try {
          const captions = await captionImage(img.file);
          setGeneratedCaptions((prev) => ({ ...prev, [img.id]: captions }));
          done++;
          setCurrentFile(`Captioning: ${img.name} (${done}/${analyzedImages.length})`);
          setAnalysisProgress(Math.round((done / analyzedImages.length) * 100));
        } catch { /* skip failed */ }
      }
      toast.success(`Generated captions for ${done} images`);
      pendo.track("batch_captions_generated", {
        totalImages: analyzedImages.length,
        successCount: done,
        failedCount: analyzedImages.length - done,
      });
    } finally {
      setCaptionGenerating(false);
      setCurrentFile("");
      setAnalysisProgress(0);
    }
  }, [analyzedImages]);

  const removeUpload = useCallback((id: string) => setUploadedImages((p) => p.filter((i) => i.id !== id)), []);

  // ── Computed ──────────────────────────────────────────────────────────────

  const filteredImages = analyzedImages.filter((img) => {
    const mf = galleryFilter === "all" || img.category === galleryFilter;
    const ms =
      !searchQuery ||
      img.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (captions[img.id] ?? img.caption).toLowerCase().includes(searchQuery.toLowerCase());
    return mf && ms;
  });

  const counts = {
    all: analyzedImages.length,
    keeper: analyzedImages.filter((a) => a.category === "keeper").length,
    duplicate: analyzedImages.filter((a) => a.category === "duplicate").length,
    blurry: analyzedImages.filter((a) => a.category === "blurry").length,
    rejected: analyzedImages.filter((a) => a.category === "rejected").length,
  };

  const objectFilteredImages = analyzedImages.filter((img) => {
    const nameMatch = !objectSearch || img.name.toLowerCase().includes(objectSearch.toLowerCase());
    const promptMatch = !objectFilterIds || objectFilterIds.includes(img.id);
    return nameMatch && promptMatch;
  });

  const downloadZip = useCallback(async () => {
    if (!filteredImages.length) { toast.error("No images to export"); return; }

    const filterLabel = galleryFilter === "all" ? "all" : galleryFilter;
    const zipName = galleryFilter === "all"
      ? `${projectTitle || "album"}_all`
      : `${projectTitle || "album"}_${filterLabel}`;

    const downloadPromise = downloadSessionZip(
      sessionId || "local",
      zipName,
      filteredImages.map((k) => ({ filename: k.name, url: k.url }))
    ).then(() => {
      pendo.track("gallery_exported_zip", {
        filterCategory: filterLabel,
        imageCount: filteredImages.length,
        projectTitle: (projectTitle || "album").substring(0, 100),
        sessionId: sessionId || "local",
        zipFilename: zipName,
      });
    });
    toast.promise(downloadPromise, {
      loading: `Packaging ${filteredImages.length} ${filterLabel} images…`,
      success: "ZIP download started",
      error: "Download failed",
    });
  }, [sessionId, projectTitle, filteredImages, galleryFilter]);

  useEffect(() => {
    if (!searchQuery.trim() || analysisState !== "complete") return;
    const timer = setTimeout(() => {
      pendo.track("gallery_search_executed", {
        searchQuery: searchQuery.trim().substring(0, 100),
        resultsCount: filteredImages.length,
        totalImages: analyzedImages.length,
        activeFilter: galleryFilter,
      });
    }, 500);
    return () => clearTimeout(timer);
  }, [searchQuery, analysisState, filteredImages.length, analyzedImages.length, galleryFilter]);

  // ─────────────────────────────────────────────────────────────────────────
  // HOME SCREEN
  // ─────────────────────────────────────────────────────────────────────────

  if (screen === "home") return (
    <div className="min-h-screen bg-background text-foreground" style={{ fontFamily: "'Inter', sans-serif" }}>
      <Toaster position="top-right" theme="dark" />

      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-xl px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-primary/15 border border-primary/25 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <span className="font-semibold text-base tracking-tight">CuratorAI</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground px-3 py-1.5 hidden md:block">For photographers &amp; studios</span>
          <button
            onClick={() => setScreen("app")}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold hover:opacity-90 transition-all"
          >
            Launch App
          </button>
        </div>
      </nav>

      {/* Hero */}
      <div className="max-w-5xl mx-auto px-8 pt-20 pb-12 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-mono mb-8">
          <Zap className="w-3 h-3" />
          Powered by Computer Vision AI
        </div>
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-5 leading-[1.1]">
          Your AI photo
          <br />
          <span className="text-primary">curator &amp; analyst</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-xl mx-auto mb-10 leading-relaxed">
          Upload your event or album photos and instantly filter duplicates, detect blurry shots, reject blink captures, and identify objects on demand.
        </p>
        <div className="flex items-center justify-center gap-3 flex-wrap">
          <button
            onClick={() => { setScreen("app"); setActiveTab("dashboard"); }}
            className="flex items-center gap-2 px-7 py-3.5 bg-primary text-primary-foreground rounded-xl font-semibold text-sm hover:opacity-90 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <Play className="w-4 h-4" />
            Start Curating Free
          </button>
        </div>
      </div>

      {/* Stats bar */}
      <div className="max-w-5xl mx-auto px-8 mb-14">
        <div className="grid grid-cols-3 gap-4">
          {[
            { value: "10x", label: "Faster than manual curation" },
            { value: "4", label: "Smart filter categories" },
            { value: "99%", label: "Detection accuracy" },
          ].map((s) => (
            <div key={s.label} className="text-center p-5 bg-card border border-border rounded-2xl">
              <div className="text-3xl font-bold text-primary font-mono mb-1">{s.value}</div>
              <div className="text-xs text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature cards */}
      <div className="max-w-5xl mx-auto px-8 mb-14">
        <h2 className="text-xl font-bold mb-5 tracking-tight">Four intelligent filters</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {(["keeper", "duplicate", "blurry", "rejected"] as const).map((cat) => {
            const cfg = CATEGORY_CONFIG[cat];
            const Icon = cfg.icon;
            const descs: Record<Category, string> = {
              keeper: "Best quality shots automatically selected and preserved",
              duplicate: "Near-identical frames detected, grouped and separated",
              blurry: "Out-of-focus and motion-blur shots flagged for removal",
              rejected: "Blinking faces and low-quality shots filtered out",
            };
            const stats: Record<Category, string> = { keeper: "50–70%", duplicate: "15–30%", blurry: "10–20%", rejected: "~5%" };
            return (
              <div key={cat} className={`p-5 rounded-2xl ${cfg.bg} border ${cfg.border} hover:scale-[1.02] transition-transform cursor-default`}>
                <Icon className={`w-5 h-5 ${cfg.color} mb-3`} />
                <div className="text-sm font-semibold mb-1.5">{cfg.label}</div>
                <div className="text-xs text-muted-foreground mb-4 leading-relaxed">{descs[cat]}</div>
                <div className={`text-xs font-mono font-bold ${cfg.color}`}>{stats[cat]} typical</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Object detection section */}
      <div className="max-w-5xl mx-auto px-8 mb-16">
        <div className="bg-card border border-border rounded-2xl p-8 flex flex-col md:flex-row items-center gap-8">
          <div className="flex-1">
            <div className="inline-flex items-center gap-1.5 text-xs font-mono text-primary mb-3">
              <Target className="w-3 h-3" />
              Object Detection Engine
            </div>
            <h2 className="text-2xl font-bold mb-3 tracking-tight">See what is in every photo</h2>
            <p className="text-sm text-muted-foreground leading-relaxed mb-6">
              Every image can be searched with natural language via Grounding DINO (ONNX WASM) — try prompts like &quot;person with green hat&quot; to filter your album client-side.
            </p>
            <button
              onClick={() => { setScreen("app"); setActiveTab("objects"); }}
              className="flex items-center gap-1.5 text-sm font-medium text-primary hover:underline underline-offset-2"
            >
              Try Object Detection <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex-shrink-0 w-56 h-44 rounded-xl border border-border bg-muted/50 relative overflow-hidden flex items-center justify-center">
            <Camera className="w-12 h-12 text-muted-foreground/15" />
            {[
              { t: "14%", l: "16%", w: "32%", h: "50%", label: "Person", conf: "96" },
              { t: "42%", l: "56%", w: "30%", h: "32%", label: "Car", conf: "88" },
              { t: "8%", l: "60%", w: "22%", h: "25%", label: "Tree", conf: "79" },
            ].map((b, i) => (
              <div
                key={i}
                className="absolute border border-primary/60 rounded-sm"
                style={{ top: b.t, left: b.l, width: b.w, height: b.h }}
              >
                <span className="absolute -top-4 left-0 whitespace-nowrap text-[9px] font-mono bg-primary/85 text-white px-1.5 py-0.5 rounded-t-sm">
                  {b.label} {b.conf}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="max-w-5xl mx-auto px-8 border-t border-border/40 py-7 flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-primary" />
          <span>CuratorAI</span>
        </div>
        <div className="flex items-center gap-5">
          <span className="hover:text-foreground cursor-pointer transition-colors">Privacy</span>
          <span className="hover:text-foreground cursor-pointer transition-colors">Terms</span>
          <span>© 2025</span>
        </div>
      </div>
    </div>
  );

  // ─────────────────────────────────────────────────────────────────────────
  // MAIN APP
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden" style={{ fontFamily: "'Inter', sans-serif" }}>
      <Toaster position="top-right" theme="dark" />

      {/* App Header */}
      <header className="flex-shrink-0 border-b border-border bg-background/90 backdrop-blur-xl px-6 h-13 flex items-center justify-between z-10">
        <button onClick={() => setScreen("home")} className="flex items-center gap-2.5 group">
          <div className="w-7 h-7 rounded-lg bg-primary/15 border border-primary/25 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-primary" />
          </div>
          <span className="font-semibold text-sm tracking-tight group-hover:text-primary transition-colors">CuratorAI</span>
        </button>

        {/* Tab switcher */}
        <div className="flex items-center gap-0.5 bg-muted/60 p-1 rounded-xl border border-border/60">
          {(
            [
              { id: "dashboard" as Tab, icon: LayoutDashboard, label: "Dashboard" },
              { id: "gallery" as Tab, icon: Image, label: "Gallery" },
              { id: "objects" as Tab, icon: Target, label: "Objects" },
              { id: "caption" as Tab, icon: MessageSquare, label: "Captions" },
            ] as const
          ).map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === id
                  ? "bg-background text-foreground shadow-sm border border-border/50"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
              {id === "gallery" && counts.all > 0 && (
                <span className="px-1.5 py-0.5 text-[10px] font-mono bg-primary/25 text-primary rounded-full leading-none">
                  {counts.all}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          {analysisState === "complete" && (
            <div className="hidden md:flex items-center gap-1.5 text-xs text-emerald-400 font-mono">
              <CheckCircle className="w-3.5 h-3.5" />
              {counts.all} analyzed
            </div>
          )}
          {projectTitle && (
            <div className="hidden lg:flex items-center gap-1.5 text-xs text-muted-foreground border border-border/50 rounded-lg px-2.5 py-1">
              <Tag className="w-3 h-3" />
              <span className="max-w-32 truncate">{projectTitle}</span>
            </div>
          )}
          <button
            onClick={() => setScreen("home")}
            className="p-2 hover:bg-accent rounded-lg transition-colors text-muted-foreground hover:text-foreground"
            title="Home"
          >
            <Home className="w-4 h-4" />
          </button>
          <button
            onClick={async () => {
              try {
                const res = await fetch(`${AI_CONFIG.STORAGE_API}/api/sessions/cleanup`, { method: "POST" });
                const data = await res.json();
                setCleanupCount(data.deleted);
                toast.success(`Cleaned ${data.deleted} old session(s) (>72h)`);
                pendo.track("old_sessions_cleaned", {
                  deletedSessionCount: data.deleted,
                });
                setTimeout(() => setCleanupCount(null), 3000);
              } catch { /* ignore */ }
            }}
            className="p-2 hover:bg-accent rounded-lg transition-colors text-muted-foreground hover:text-foreground"
            title="Clean old sessions (>72h)"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">

          {/* ── DASHBOARD ── */}
          {activeTab === "dashboard" && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="h-full overflow-y-auto"
            >
              <div className="max-w-5xl mx-auto px-6 py-8">
                <div className="mb-8">
                  <h1 className="text-2xl font-bold tracking-tight">New Project</h1>
                  <p className="text-sm text-muted-foreground mt-1">Configure your album and upload images for analysis</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
                  {/* ── Left: Project Setup ── */}
                  <div className="lg:col-span-2 flex flex-col gap-5">

                    {/* Project Details */}
                    <div className="bg-card border border-border rounded-2xl p-5">
                      <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
                        <Tag className="w-4 h-4 text-primary" />
                        Project Details
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Album / Event Title</label>
                          <input
                            type="text"
                            value={projectTitle}
                            onChange={(e) => setProjectTitle(e.target.value)}
                            placeholder="e.g. Sarah & Mark Wedding 2025"
                            className="w-full bg-muted/50 border border-border rounded-xl px-3 py-2.5 text-sm placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/40 transition-all"
                          />
                        </div>
                        <div>
                          <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Event Date</label>
                          <div className="relative">
                            <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                            <input
                              type="date"
                              value={eventDate}
                              onChange={(e) => setEventDate(e.target.value)}
                              className="w-full bg-muted/50 border border-border rounded-xl pl-10 pr-3 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/40 transition-all"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Analysis Pipeline Info */}
                    <div className="bg-card border border-border rounded-2xl p-5 flex-1">
                      <h3 className="text-sm font-semibold mb-0.5 flex items-center gap-2">
                        <Zap className="w-4 h-4 text-primary" />
                        Analysis Pipeline
                      </h3>
                      <p className="text-xs text-muted-foreground mb-4">
                        Browser-side AI pipeline (WASM)
                      </p>
                      <div className="space-y-2.5 text-xs text-muted-foreground">
                        {[
                          "TensorFlow.js CNN IQA blur regressor",
                          "MediaPipe Face Landmarker blink detection",
                          "ONNX DINOv2 burst grouping (>0.90 similarity)",
                          "Grounding DINO natural language object search",
                        ].map((step) => (
                          <div key={step} className="flex items-center gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0" />
                            {step}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* ── Right: Upload & Analyze ── */}
                  <div className="lg:col-span-3 flex flex-col gap-5">

                    {/* Upload zone */}
                    <div className="bg-card border border-border rounded-2xl p-5">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                          <Upload className="w-4 h-4 text-primary" />
                          Upload Images
                          {uploadedImages.length > 0 && (
                            <span className="px-2 py-0.5 text-xs font-mono bg-primary/20 text-primary rounded-full">
                              {uploadedImages.length}
                            </span>
                          )}
                        </h3>
                        <div className="flex gap-1.5">
                          <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => processFiles(Array.from(e.target.files ?? []), "file_picker")}
                          />
                          <input
                            ref={folderInputRef}
                            {...folderInputProps}
                            type="file"
                            multiple
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => processFiles(Array.from(e.target.files ?? []), "folder_picker")}
                          />
                          <button
                            onClick={() => fileInputRef.current?.click()}
                            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-border rounded-lg hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
                          >
                            <Upload className="w-3.5 h-3.5" /> Add Files
                          </button>
                          <button
                            onClick={() => folderInputRef.current?.click()}
                            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-border rounded-lg hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
                          >
                            <FolderOpen className="w-3.5 h-3.5" /> Add Folder
                          </button>
                          {uploadedImages.length > 0 && (
                            <button
                              onClick={() => { pendo.track("upload_list_cleared", { clearedImageCount: uploadedImages.length }); setUploadedImages([]); }}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-destructive/30 rounded-lg hover:bg-destructive/10 transition-colors text-destructive/70 hover:text-destructive"
                            >
                              <Trash2 className="w-3.5 h-3.5" /> Clear
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Drop zone */}
                      <div
                        onDrop={handleDrop}
                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onClick={() => fileInputRef.current?.click()}
                        className={`cursor-pointer rounded-xl border-2 border-dashed transition-all py-9 flex flex-col items-center justify-center gap-3 ${
                          isDragging
                            ? "border-primary/60 bg-primary/5"
                            : "border-border/50 hover:border-primary/35 hover:bg-primary/3"
                        }`}
                      >
                        <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-colors ${isDragging ? "bg-primary/20" : "bg-muted"}`}>
                          <Upload className={`w-5 h-5 transition-colors ${isDragging ? "text-primary" : "text-muted-foreground/50"}`} />
                        </div>
                        <div className="text-center">
                          <div className="text-sm font-medium mb-0.5">
                            {isDragging ? "Release to upload" : "Drop images or folders here"}
                          </div>
                          <div className="text-xs text-muted-foreground">JPG, PNG, HEIC, WEBP, RAW supported</div>
                        </div>
                      </div>

                      {/* File list */}
                      {uploadedImages.length > 0 && (
                        <div className="mt-4 max-h-44 overflow-y-auto space-y-1 pr-0.5">
                          {uploadedImages.map((img) => (
                            <div key={img.id} className="flex items-center gap-3 px-2.5 py-2 rounded-xl hover:bg-muted/50 group transition-colors">
                              <div className="w-9 h-9 rounded-lg overflow-hidden bg-muted flex-shrink-0 border border-border/50">
                                <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="text-xs font-medium truncate">{img.name}</div>
                                <div className="text-[10px] text-muted-foreground font-mono">{formatSize(img.size)}</div>
                              </div>
                              <button
                                onClick={() => removeUpload(img.id)}
                                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/15 rounded-lg transition-all text-muted-foreground hover:text-destructive"
                              >
                                <X className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Analyze */}
                    <div className="bg-card border border-border rounded-2xl p-5">
                      {analysisState === "idle" && (
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <Zap className="w-4 h-4 text-primary" />
                            <span className="text-sm font-semibold">Start Analysis</span>
                          </div>
                          <p className="text-xs text-muted-foreground mb-4">
                            All AI runs in your browser via WASM — zero server RAM. Storage backend only saves files + results.json.
                          </p>
                          <button
                            onClick={startAnalysis}
                            disabled={!uploadedImages.length}
                            className="w-full flex items-center justify-center gap-2 py-3 bg-primary text-primary-foreground rounded-xl font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-35 disabled:cursor-not-allowed hover:scale-[1.01] active:scale-[0.99]"
                          >
                            <Zap className="w-4 h-4" />
                            Start Analyze
                            {uploadedImages.length > 0 && (
                              <span className="font-normal text-xs opacity-70 font-mono">
                                ({uploadedImages.length} image{uploadedImages.length !== 1 ? "s" : ""})
                              </span>
                            )}
                          </button>
                        </div>
                      )}

                      {analysisState === "analyzing" && (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-semibold">Analyzing images…</span>
                            <span className="text-sm font-mono text-primary">{analysisProgress}%</span>
                          </div>
                          <div className="h-1.5 bg-muted rounded-full overflow-hidden mb-3">
                            <motion.div
                              className="h-full bg-primary rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${analysisProgress}%` }}
                              transition={{ duration: 0.3 }}
                            />
                          </div>
                          <div className="text-xs text-muted-foreground font-mono truncate mb-3">Processing: {currentFile}</div>
                          <div className="grid grid-cols-2 gap-1.5 text-xs">
                            {(["Duplicate Detection", "Quality Assessment", "Blink Detection", "Caption Generation"] as const).map((step, i) => (
                              <div key={step} className={`flex items-center gap-1.5 ${analysisProgress > i * 25 ? "text-primary" : "text-muted-foreground/50"}`}>
                                <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${analysisProgress > i * 25 ? "bg-primary" : "bg-muted-foreground/20"}`} />
                                {step}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {analysisState === "complete" && (
                        <div>
                          <div className="flex items-center gap-2 text-emerald-400 mb-4">
                            <CheckCircle className="w-5 h-5" />
                            <span className="font-semibold text-sm">Analysis Complete</span>
                          </div>
                          <div className="grid grid-cols-2 gap-2 mb-4">
                            {(["keeper", "duplicate", "blurry", "rejected"] as const).map((cat) => {
                              const cfg = CATEGORY_CONFIG[cat];
                              const Icon = cfg.icon;
                              return (
                                <div key={cat} className={`flex items-center gap-3 p-3 rounded-xl ${cfg.bg} border ${cfg.border}`}>
                                  <span className={`text-xl font-bold font-mono ${cfg.color}`}>{counts[cat]}</span>
                                  <div>
                                    <div className={`text-xs font-medium ${cfg.color}`}>{cfg.label}</div>
                                    <Icon className={`w-3 h-3 mt-0.5 ${cfg.color} opacity-60`} />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                          <button
                            onClick={() => setActiveTab("gallery")}
                            className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary/10 text-primary border border-primary/20 rounded-xl text-sm font-medium hover:bg-primary/15 transition-colors"
                          >
                            View Gallery <ChevronRight className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* ── GALLERY ── */}
          {activeTab === "gallery" && (
            <motion.div
              key="gallery"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="h-full flex flex-col overflow-hidden"
            >
              {analysisState !== "complete" ? (
                <div className="flex-1 flex items-center justify-center flex-col gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-muted border border-border flex items-center justify-center">
                    <Image className="w-7 h-7 text-muted-foreground/30" />
                  </div>
                  <div className="text-center">
                    <div className="font-semibold mb-1">No images analyzed yet</div>
                    <div className="text-sm text-muted-foreground mb-5">Upload and analyze your images from the Dashboard tab</div>
                    <button
                      onClick={() => setActiveTab("dashboard")}
                      className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-xl text-sm font-medium mx-auto hover:opacity-90 transition-opacity"
                    >
                      Go to Dashboard <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {/* Toolbar */}
                  <div className="flex-shrink-0 border-b border-border px-5 py-2.5 flex items-center gap-2.5 flex-wrap">
                    {/* Filter pills */}
                    <div className="flex items-center gap-1 flex-1 overflow-x-auto min-w-0">
                      <button
                        onClick={() => setGalleryFilter("all")}
                        className={`flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                          galleryFilter === "all"
                            ? "bg-primary/15 text-primary border-primary/25"
                            : "border-transparent text-muted-foreground hover:text-foreground hover:bg-accent"
                        }`}
                      >
                        All <span className="font-mono">{counts.all}</span>
                      </button>
                      {(["keeper", "duplicate", "blurry", "rejected"] as const).map((cat) => {
                        const cfg = CATEGORY_CONFIG[cat];
                        const Icon = cfg.icon;
                        return (
                          <button
                            key={cat}
                            onClick={() => setGalleryFilter(cat)}
                            className={`flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                              galleryFilter === cat
                                ? `${cfg.bg} ${cfg.color} ${cfg.border}`
                                : "border-transparent text-muted-foreground hover:text-foreground hover:bg-accent"
                            }`}
                          >
                            <Icon className="w-3.5 h-3.5" />
                            {cfg.label}
                            <span className="font-mono">{counts[cat]}</span>
                          </button>
                        );
                      })}
                    </div>

                    {/* Search */}
                    <div className="relative flex-shrink-0">
                      <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search…"
                        className="bg-muted/50 border border-border rounded-lg pl-8 pr-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-primary/50 w-36 placeholder:text-muted-foreground/40"
                      />
                    </div>

                    {/* View toggle */}
                    <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5 flex-shrink-0 border border-border/50">
                      <button
                        onClick={() => setViewMode("grid")}
                        className={`p-1.5 rounded transition-colors ${viewMode === "grid" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                      >
                        <LayoutGrid className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => setViewMode("list")}
                        className={`p-1.5 rounded transition-colors ${viewMode === "list" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                      >
                        <List className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {/* Export */}
                    <button
                      onClick={downloadZip}
                      className="flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 bg-primary/10 text-primary border border-primary/20 rounded-lg text-xs font-medium hover:bg-primary/20 transition-colors"
                    >
                      <Archive className="w-3.5 h-3.5" />
                      Export ZIP
                      <span className="font-mono opacity-70">({filteredImages.length})</span>
                    </button>
                  </div>

                  {/* Image grid / list */}
                  <div className="flex-1 overflow-y-auto px-5 py-5">
                    {filteredImages.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                        <Filter className="w-8 h-8 opacity-25" />
                        <span className="text-sm">No images match this filter</span>
                      </div>
                    ) : viewMode === "grid" ? (
                      <div className="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-3 space-y-3">
                        {filteredImages.map((img) => {
                          const cfg = CATEGORY_CONFIG[img.category];
                          const Icon = cfg.icon;
                          const caption = captions[img.id] ?? img.caption;
                          const isSelected = selectedImage?.id === img.id;
                          return (
                            <div
                              key={img.id}
                              onClick={() => setSelectedImage(img)}
                              className={`break-inside-avoid bg-card border rounded-xl overflow-hidden transition-all group cursor-pointer hover:border-primary/30 ${isSelected ? "border-primary/50 ring-1 ring-primary/20" : "border-border"}`}
                            >
                              <div className="relative overflow-hidden bg-muted" style={{ aspectRatio: "4/3" }}>
                                <img
                                  src={img.url}
                                  alt={img.name}
                                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                />
                                <div className={`absolute top-2 left-2 flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium ${cfg.badgeBg} backdrop-blur-sm`}>
                                  <Icon className="w-2.5 h-2.5" />
                                  {cfg.label}
                                </div>
                                <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-mono bg-black/55 text-white backdrop-blur-sm">
                                  {img.confidence}%
                                </div>
                                <button
                                  onClick={() => { setSelectedImage(img); setActiveTab("objects"); }}
                                  className="absolute bottom-2 right-2 p-1.5 bg-black/65 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                                  title="View detected objects"
                                >
                                  <Target className="w-3 h-3 text-white" />
                                </button>
                              </div>
                              <div className="p-3">
                                <div className="text-[11px] font-medium truncate text-muted-foreground mb-1.5">{img.name}</div>
                                {editingCaption === img.id ? (
                                  <div className="flex gap-1">
                                    <input
                                      autoFocus
                                      value={editValue}
                                      onChange={(e) => setEditValue(e.target.value)}
                                      className="flex-1 text-xs bg-muted border border-border rounded-lg px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
                                      onKeyDown={(e) => {
                                        if (e.key === "Enter") { setCaptions((p) => ({ ...p, [img.id]: editValue })); setEditingCaption(null); pendo.track("caption_edited", { imageId: img.id, imageName: img.name, imageCategory: img.category, captionLength: editValue.length }); }
                                        if (e.key === "Escape") setEditingCaption(null);
                                      }}
                                    />
                                    <button
                                      onClick={() => { setCaptions((p) => ({ ...p, [img.id]: editValue })); setEditingCaption(null); pendo.track("caption_edited", { imageId: img.id, imageName: img.name, imageCategory: img.category, captionLength: editValue.length }); }}
                                      className="px-2 py-1 bg-primary/20 text-primary rounded-lg text-[10px] font-medium hover:bg-primary/30 transition-colors"
                                    >
                                      Save
                                    </button>
                                  </div>
                                ) : (
                                  <div
                                    onClick={() => { setEditingCaption(img.id); setEditValue(caption); }}
                                    className="text-xs text-muted-foreground cursor-text hover:text-foreground transition-colors line-clamp-2 leading-relaxed"
                                    title="Click to edit caption"
                                  >
                                    {caption}
                                  </div>
                                )}
                                <div className="flex gap-1.5 mt-2">
                                  <button
                                    onClick={(e) => { e.stopPropagation(); runCaptionSingle(img); }}
                                    disabled={captionGenerating}
                                    className="flex items-center gap-1 px-2 py-1 bg-primary/10 text-primary border border-primary/20 rounded-md text-[10px] font-medium hover:bg-primary/15 transition-colors disabled:opacity-40"
                                  >
                                    <MessageSquare className="w-2.5 h-2.5" />
                                    {captionGenerating ? "..." : "Caption"}
                                  </button>
                                  <button
                                    onClick={(e) => { e.stopPropagation(); setSelectedImage(img); setActiveTab("objects"); }}
                                    className="flex items-center gap-1 px-2 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-md text-[10px] font-medium hover:bg-emerald-500/15 transition-colors"
                                  >
                                    <Scan className="w-2.5 h-2.5" />
                                    Detect
                                  </button>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="space-y-1.5">
                        {filteredImages.map((img) => {
                          const cfg = CATEGORY_CONFIG[img.category];
                          const Icon = cfg.icon;
                          const caption = captions[img.id] ?? img.caption;
                          return (
                            <div
                              key={img.id}
                              className="flex items-center gap-4 px-4 py-3 bg-card border border-border rounded-xl hover:border-primary/30 transition-all group"
                            >
                              <div className="w-14 h-10 rounded-lg overflow-hidden bg-muted flex-shrink-0 border border-border/50">
                                <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="text-sm font-medium truncate">{img.name}</div>
                                {editingCaption === img.id ? (
                                  <div className="flex gap-1 mt-1">
                                    <input
                                      autoFocus
                                      value={editValue}
                                      onChange={(e) => setEditValue(e.target.value)}
                                      className="flex-1 text-xs bg-muted border border-border rounded px-2 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/50"
                                      onKeyDown={(e) => {
                                        if (e.key === "Enter") { setCaptions((p) => ({ ...p, [img.id]: editValue })); setEditingCaption(null); pendo.track("caption_edited", { imageId: img.id, imageName: img.name, imageCategory: img.category, captionLength: editValue.length }); }
                                        if (e.key === "Escape") setEditingCaption(null);
                                      }}
                                    />
                                    <button onClick={() => { setCaptions((p) => ({ ...p, [img.id]: editValue })); setEditingCaption(null); pendo.track("caption_edited", { imageId: img.id, imageName: img.name, imageCategory: img.category, captionLength: editValue.length }); }} className="px-2 py-0.5 bg-primary/20 text-primary rounded text-[10px]">Save</button>
                                  </div>
                                ) : (
                                  <div
                                    onClick={() => { setEditingCaption(img.id); setEditValue(caption); }}
                                    className="text-xs text-muted-foreground truncate mt-0.5 cursor-text hover:text-foreground transition-colors"
                                  >
                                    {caption}
                                  </div>
                                )}
                                <div className="flex gap-1.5 mt-1.5">
                                  <button
                                    onClick={(e) => { e.stopPropagation(); runCaptionSingle(img); }}
                                    disabled={captionGenerating}
                                    className="flex items-center gap-1 px-2 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded text-[10px] font-medium hover:bg-primary/15 transition-colors disabled:opacity-40"
                                  >
                                    <MessageSquare className="w-2.5 h-2.5" />
                                    {captionGenerating ? "..." : "Caption"}
                                  </button>
                                  <button
                                    onClick={(e) => { e.stopPropagation(); setSelectedImage(img); setActiveTab("objects"); }}
                                    className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded text-[10px] font-medium hover:bg-emerald-500/15 transition-colors"
                                  >
                                    <Scan className="w-2.5 h-2.5" />
                                    Detect
                                  </button>
                                </div>
                              </div>
                              <div className={`flex-shrink-0 flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium ${cfg.badgeBg}`}>
                                <Icon className="w-3 h-3" />
                                {cfg.label}
                              </div>
                              <div className="text-xs font-mono text-muted-foreground flex-shrink-0">{img.confidence}%</div>
                              <div className="text-xs font-mono text-muted-foreground flex-shrink-0 hidden sm:block">{formatSize(img.size)}</div>
                              <button
                                onClick={() => { setSelectedImage(img); setActiveTab("objects"); }}
                                className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-accent rounded-lg transition-all flex-shrink-0"
                                title="View objects"
                              >
                                <Target className="w-4 h-4 text-muted-foreground" />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Selected image detail overlay */}
                    <AnimatePresence>
                      {selectedImage && filteredImages.some(f => f.id === selectedImage.id) && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8"
                          onClick={() => setSelectedImage(null)}
                        >
                          <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.95, opacity: 0 }}
                            className="relative max-w-4xl w-full max-h-full bg-card border border-border rounded-2xl overflow-hidden shadow-2xl flex flex-col"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <div className="flex items-center justify-between px-5 py-3 border-b border-border">
                              <div>
                                <div className="text-sm font-semibold">{selectedImage.name}</div>
                                <div className="flex items-center gap-2 mt-0.5">
                                  <span className={`text-[10px] font-medium ${CATEGORY_CONFIG[selectedImage.category].color}`}>
                                    {CATEGORY_CONFIG[selectedImage.category].label}
                                  </span>
                                  <span className="text-[10px] text-muted-foreground font-mono">{selectedImage.confidence}%</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => runCaptionSingle(selectedImage)}
                                  disabled={captionGenerating}
                                  className="flex items-center gap-1.5 px-3 py-1.5 bg-primary/15 text-primary border border-primary/25 rounded-lg text-xs font-semibold hover:bg-primary/20 transition-all disabled:opacity-40"
                                >
                                  <MessageSquare className="w-3.5 h-3.5" />
                                  {captionGenerating ? "Generating..." : "Generate Caption"}
                                </button>
                                <button
                                  onClick={() => { setSelectedImage(selectedImage); setActiveTab("objects"); }}
                                  className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 rounded-lg text-xs font-semibold hover:bg-emerald-500/20 transition-all"
                                >
                                  <Scan className="w-3.5 h-3.5" />
                                  Detect Objects
                                </button>
                                <button onClick={() => setSelectedImage(null)} className="p-1.5 hover:bg-accent rounded-lg transition-colors">
                                  <X className="w-4 h-4" />
                                </button>
                              </div>
                            </div>
                            <div className="flex-1 overflow-auto p-5 flex gap-5">
                              <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-xl overflow-hidden">
                                <img src={selectedImage.url} alt={selectedImage.name} className="max-w-full max-h-[60vh] object-contain rounded-lg" />
                              </div>
                              <div className="w-72 flex-shrink-0 space-y-4">
                                <div>
                                  <div className="text-[10px] font-semibold text-muted-foreground tracking-wider mb-2">CAPTION</div>
                                  {(() => {
                                    const genCaps = generatedCaptions[selectedImage.id];
                                    const editCap = captions[selectedImage.id] ?? selectedImage.caption;
                                    return genCaps && genCaps.length > 0 ? (
                                      <div className="space-y-1.5">
                                        {genCaps.map((c, i) => (
                                          <div key={i} className="text-xs text-foreground leading-relaxed p-2 bg-background/50 rounded-lg border border-border/50">
                                            <span className="text-[10px] font-mono text-primary mr-1.5">#{i + 1}</span>
                                            {c}
                                          </div>
                                        ))}
                                      </div>
                                    ) : (
                                      <div className="text-xs text-muted-foreground leading-relaxed p-2 bg-background/50 rounded-lg border border-border/50">
                                        {editCap || "No caption yet — click Generate Caption"}
                                      </div>
                                    );
                                  })()}
                                </div>
                                <div>
                                  <div className="text-[10px] font-semibold text-muted-foreground tracking-wider mb-2">OBJECTS</div>
                                  {selectedImage.objects.length > 0 ? (
                                    <div className="space-y-1">
                                      {selectedImage.objects.map((obj, i) => (
                                        <div key={i} className="flex items-center justify-between text-xs p-1.5 bg-background/50 rounded-lg border border-border/50">
                                          <span className="font-medium">{obj.label}</span>
                                          <span className="font-mono text-primary">{obj.confidence}%</span>
                                        </div>
                                      ))}
                                    </div>
                                  ) : (
                                    <div className="text-xs text-muted-foreground/50 italic">Not scanned — click Detect Objects</div>
                                  )}
                                </div>
                                <div>
                                  <div className="text-[10px] font-semibold text-muted-foreground tracking-wider mb-2">INFO</div>
                                  <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs">
                                      <span className="text-muted-foreground">Quality</span>
                                      <span className="font-mono">{selectedImage.confidence}%</span>
                                    </div>
                                    <div className="flex justify-between text-xs">
                                      <span className="text-muted-foreground">Size</span>
                                      <span className="font-mono">{formatSize(selectedImage.size)}</span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </>
              )}
            </motion.div>
          )}

          {/* ── OBJECTS ── */}
          {activeTab === "objects" && (
            <motion.div
              key="objects"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="h-full flex overflow-hidden"
            >
              {analysisState !== "complete" ? (
                <div className="flex-1 flex items-center justify-center flex-col gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-muted border border-border flex items-center justify-center">
                    <Target className="w-7 h-7 text-muted-foreground/30" />
                  </div>
                  <div className="text-center">
                    <div className="font-semibold mb-1">No detection data</div>
                    <div className="text-sm text-muted-foreground mb-5">Analyze your images first to see object detection results</div>
                    <button
                      onClick={() => setActiveTab("dashboard")}
                      className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-xl text-sm font-medium mx-auto hover:opacity-90 transition-opacity"
                    >
                      Go to Dashboard <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {/* Image selector sidebar */}
                  <div className="w-52 flex-shrink-0 border-r border-border flex flex-col overflow-hidden bg-card/50">
                    <div className="p-3 border-b border-border space-y-2">
                      <div className="text-[10px] font-semibold text-muted-foreground tracking-wider">NATURAL LANGUAGE SEARCH</div>
                      <input
                        value={nlPrompt}
                        onChange={(e) => setNlPrompt(e.target.value)}
                        placeholder="e.g. man with green hat"
                        className="w-full px-2 py-1.5 bg-muted/50 border border-border rounded-lg text-xs focus:outline-none focus:ring-1 focus:ring-primary/50"
                      />
                      <button
                        onClick={runGlobalObjectSearch}
                        disabled={objectDetectingId === "all"}
                        className="w-full py-1.5 bg-primary/15 text-primary border border-primary/25 rounded-lg text-xs font-medium hover:bg-primary/20 disabled:opacity-40"
                      >
                        {objectDetectingId === "all" ? "Searching…" : "Search All Images"}
                      </button>
                      <div className="text-[10px] font-semibold text-muted-foreground tracking-wider pt-1">SELECT IMAGE</div>
                      <div className="relative">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground pointer-events-none" />
                        <input
                          value={objectSearch}
                          onChange={(e) => setObjectSearch(e.target.value)}
                          placeholder="Search…"
                          className="w-full pl-7 pr-2 py-1.5 bg-muted/50 border border-border rounded-lg text-xs focus:outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-muted-foreground/40"
                        />
                      </div>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
                      {objectFilteredImages.map((img) => {
                        const cfg = CATEGORY_CONFIG[img.category];
                        return (
                          <button
                            key={img.id}
                            onClick={() => handleSelectImage(img)}
                            className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-xl text-left transition-all ${
                              selectedImage?.id === img.id
                                ? "bg-primary/10 border border-primary/20"
                                : "hover:bg-muted/50 border border-transparent"
                            }`}
                          >
                            <div className="w-10 h-7 rounded-md overflow-hidden bg-muted flex-shrink-0 border border-border/40">
                              <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-[11px] font-medium truncate">{img.name}</div>
                              <div className={`text-[10px] font-mono ${cfg.color}`}>
                                {img.objectsLoaded ? `${img.objects.length} obj` : "not scanned"}
                              </div>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Main detection area */}
                  <div className="flex-1 flex flex-col overflow-hidden">
                    {selectedImage ? (
                      <>
                        {/* Image header */}
                        <div className="flex-shrink-0 border-b border-border px-5 py-3 flex items-center justify-between bg-card/30">
                          <div className="flex items-center gap-3">
                            <div>
                              <div className="text-sm font-semibold">{selectedImage.name}</div>
                              <div className="flex items-center gap-2 mt-0.5">
                                <span className={`text-[10px] font-medium ${CATEGORY_CONFIG[selectedImage.category].color}`}>
                                  {CATEGORY_CONFIG[selectedImage.category].label}
                                </span>
                                <span className="text-muted-foreground/40 text-[10px]">·</span>
                                <span className="text-[10px] text-muted-foreground font-mono">
                                  {selectedImage.objectsLoaded
                                    ? `${selectedImage.objects.length} objects detected`
                                    : "Click Detect to scan"}
                                </span>
                                <span className="text-muted-foreground/40 text-[10px]">·</span>
                                <span className="text-[10px] text-muted-foreground font-mono">{formatSize(selectedImage.size)}</span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-xs text-muted-foreground">Bounding boxes</span>
                            <button
                              onClick={() => setShowBboxes((b) => !b)}
                              className={`relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${showBboxes ? "bg-primary" : "bg-muted"}`}
                            >
                              <div
                                className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-all duration-150 ${showBboxes ? "left-[18px]" : "left-0.5"}`}
                              />
                            </button>
                          </div>
                        </div>

                        <div className="flex-1 flex overflow-hidden">
                          {/* Image with bboxes */}
                          <div className="flex-1 overflow-hidden flex items-center justify-center p-6 bg-muted/10">
                            <div className="relative max-w-full max-h-full" style={{ aspectRatio: "4/3", maxHeight: "calc(100vh - 220px)" }}>
                              <img
                                src={selectedImage.url}
                                alt={selectedImage.name}
                                className="w-full h-full object-contain rounded-xl border border-border shadow-lg"
                              />
                              {objectDetectingId === selectedImage.id && (
                                <div className="absolute inset-0 bg-black/50 rounded-xl flex flex-col items-center justify-center gap-2">
                                  <motion.div
                                    className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full"
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                  />
                                  <span className="text-xs text-white font-medium">Running YOLO-World detection…</span>
                                </div>
                              )}
                              {/* Floating detect button - bottom center */}
                              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10">
                                <button
                                  onClick={() => runDetectSingle(selectedImage)}
                                  disabled={detectingImageId === selectedImage.id}
                                  className="flex items-center gap-2 px-5 py-2.5 bg-emerald-500 text-white rounded-xl text-sm font-semibold shadow-lg shadow-emerald-500/30 hover:bg-emerald-400 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                  <Scan className="w-4 h-4" />
                                  {detectingImageId === selectedImage.id ? "Detecting…" : "Detect Objects"}
                                </button>
                              </div>
                              <AnimatePresence>
                                {showBboxes &&
                                  selectedImage.objects.map((obj, i) => (
                                    <motion.div
                                      key={`${selectedImage.id}-${i}`}
                                      initial={{ opacity: 0, scale: 0.95 }}
                                      animate={{ opacity: 1, scale: 1 }}
                                      exit={{ opacity: 0 }}
                                      transition={{ duration: 0.15, delay: i * 0.04 }}
                                      className="absolute border-2 border-primary/70 rounded"
                                      style={{
                                        left: `${obj.bbox.x}%`,
                                        top: `${obj.bbox.y}%`,
                                        width: `${obj.bbox.w}%`,
                                        height: `${obj.bbox.h}%`,
                                      }}
                                    >
                                      <div className="absolute -top-5 left-0 whitespace-nowrap text-[9px] font-mono font-semibold bg-primary text-primary-foreground px-1.5 py-0.5 rounded-t-sm leading-tight">
                                        {obj.label} {obj.confidence}%
                                      </div>
                                    </motion.div>
                                  ))}
                              </AnimatePresence>
                            </div>
                          </div>

                          {/* Objects panel */}
                          <div className="w-60 flex-shrink-0 border-l border-border flex flex-col overflow-hidden bg-card/50">
                            <div className="p-4 border-b border-border">
                              <div className="text-[10px] font-semibold text-muted-foreground tracking-wider mb-1">DETECTED OBJECTS</div>
                              <div className="text-3xl font-bold font-mono text-primary leading-none">{selectedImage.objects.length}</div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
                              {!selectedImage.objectsLoaded ? (
                                <div className="text-xs text-muted-foreground text-center py-8">
                                  <p className="mb-2 opacity-70">No detection results yet.</p>
                                  <p className="opacity-50">Click &quot;Detect Objects&quot; below the image to scan.</p>
                                </div>
                              ) : selectedImage.objects.length === 0 ? (
                                <div className="text-xs text-muted-foreground text-center py-8 opacity-50">No objects detected in this image</div>
                              ) : (
                                selectedImage.objects
                                  .slice()
                                  .sort((a, b) => b.confidence - a.confidence)
                                  .map((obj, i) => (
                                    <div key={i} className="p-2.5 bg-background/60 border border-border/60 rounded-xl hover:border-primary/20 transition-colors">
                                      <div className="flex items-center justify-between mb-1.5">
                                        <span className="text-xs font-semibold">{obj.label}</span>
                                        <span className="text-xs font-mono text-primary">{obj.confidence}%</span>
                                      </div>
                                      <div className="h-1 bg-muted rounded-full overflow-hidden">
                                        <motion.div
                                          className="h-full bg-primary rounded-full"
                                          initial={{ width: 0 }}
                                          animate={{ width: `${obj.confidence}%` }}
                                          transition={{ duration: 0.4, delay: i * 0.05 }}
                                        />
                                      </div>
                                    </div>
                                  ))
                              )}
                            </div>
                            <div className="border-t border-border p-4 space-y-2.5">
                              <div className="text-[10px] font-semibold text-muted-foreground tracking-wider mb-2">IMAGE STATS</div>
                              {[
                                {
                                  label: "Avg confidence",
                                  value: selectedImage.objects.length
                                    ? `${Math.round(selectedImage.objects.reduce((s, o) => s + o.confidence, 0) / selectedImage.objects.length)}%`
                                    : "—",
                                },
                                { label: "Quality score", value: `${selectedImage.confidence}%` },
                                { label: "Category", value: CATEGORY_CONFIG[selectedImage.category].label, colorClass: CATEGORY_CONFIG[selectedImage.category].color },
                                { label: "File size", value: formatSize(selectedImage.size) },
                              ].map(({ label, value, colorClass }) => (
                                <div key={label} className="flex justify-between items-center text-xs">
                                  <span className="text-muted-foreground">{label}</span>
                                  <span className={`font-mono ${colorClass ?? "text-foreground"}`}>{value}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-muted-foreground">
                        <span className="text-sm">Select an image from the sidebar</span>
                      </div>
                    )}
                  </div>
                </>
              )}
            </motion.div>
          )}

          {/* ── CAPTIONS ── */}
          {activeTab === "caption" && (
            <motion.div
              key="caption"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="h-full flex flex-col overflow-hidden"
            >
              {analysisState !== "complete" ? (
                <div className="flex-1 flex items-center justify-center flex-col gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-muted border border-border flex items-center justify-center">
                    <MessageSquare className="w-7 h-7 text-muted-foreground/30" />
                  </div>
                  <div className="text-center">
                    <div className="font-semibold mb-1">No images analyzed yet</div>
                    <div className="text-sm text-muted-foreground mb-5">Analyze your images first to generate captions</div>
                    <button onClick={() => setActiveTab("dashboard")} className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-xl text-sm font-medium mx-auto hover:opacity-90 transition-opacity">
                      Go to Dashboard <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex-shrink-0 border-b border-border px-5 py-2.5 flex items-center gap-2.5">
                    <button
                      onClick={() => selectedImage && runCaptionSingle(selectedImage)}
                      disabled={captionGenerating || !selectedImage}
                      className="flex items-center gap-1.5 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-xs font-semibold hover:opacity-90 transition-all disabled:opacity-40"
                    >
                      <MessageSquare className="w-3.5 h-3.5" />
                      {captionGenerating ? "Generating…" : "Caption Selected"}
                    </button>
                    <button
                      onClick={runCaptionAll}
                      disabled={captionGenerating}
                      className="flex items-center gap-1.5 px-4 py-2 bg-primary/10 text-primary border border-primary/20 rounded-lg text-xs font-semibold hover:bg-primary/15 transition-all disabled:opacity-40"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      {captionGenerating ? "Generating…" : "Caption All"}
                    </button>
                    {captionGenerating && (
                      <div className="flex-1">
                        <div className="h-1 bg-muted rounded-full overflow-hidden">
                          <motion.div className="h-full bg-primary rounded-full" initial={{ width: 0 }} animate={{ width: `${analysisProgress}%` }} />
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex-1 overflow-y-auto px-5 py-5">
                    {analyzedImages.length === 0 ? (
                      <div className="text-center text-muted-foreground text-sm py-10">No images to caption</div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {analyzedImages.map((img) => {
                          const caps = generatedCaptions[img.id];
                          return (
                            <div key={img.id} className="bg-card border border-border rounded-xl overflow-hidden hover:border-primary/20 transition-all">
                              <div className="aspect-video bg-muted overflow-hidden">
                                <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                              </div>
                              <div className="p-3">
                                <div className="text-xs font-medium truncate text-muted-foreground mb-2">{img.name}</div>
                                {caps ? (
                                  <div className="space-y-2">
                                    {caps.map((c, i) => (
                                      <div key={i} className="text-xs text-foreground leading-relaxed p-2 bg-background/50 rounded-lg border border-border/50">
                                        <span className="text-[10px] font-mono text-primary mr-1.5">#{i + 1}</span>
                                        {c}
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="text-xs text-muted-foreground/50 italic py-2">No captions yet — click "Caption Selected"</div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
