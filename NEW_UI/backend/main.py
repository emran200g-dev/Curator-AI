"""CuratorAI backend — in-memory image processing via Roboflow serverless API."""
import base64
import json
import logging
import os
from typing import List

import requests as http_requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_URL = "https://serverless.roboflow.com"

app = FastAPI(title="CuratorAI", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _roboflow_workflow(workspace: str, workflow: str, image_bytes: bytes, parameters: dict = None):
    """Call Roboflow serverless workflow API — pure in-memory, no disk I/O."""
    url = f"{ROBOFLOW_URL}/infer/workflows/{workspace}/{workflow}"
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    body = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": b64,
            }
        },
    }
    if parameters:
        body["parameters"] = parameters

    logger.info("Roboflow POST %s (image=%d bytes)", url, len(image_bytes))
    resp = http_requests.post(url, json=body, timeout=120)
    logger.info("Roboflow response %d", resp.status_code)
    return resp


@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects using YOLO-World — processes image in-memory from request payload."""
    try:
        contents = await file.read()
        logger.info("DETECT: %s (%d bytes)", file.filename, len(contents))

        resp = _roboflow_workflow(
            workspace="mds-workspace-l14ck",
            workflow="yolo-world-large-demo",
            image_bytes=contents,
            parameters={
                "classes": [
                    "person", "man", "woman", "boy", "girl", "child", "baby", "crowd",
                    "dog", "cat", "bird", "horse", "cow", "sheep", "pig", "rabbit", "deer",
                    "fish", "turtle", "snake", "frog", "butterfly", "bee", "squirrel", "hamster",
                    "elephant", "lion", "tiger", "bear", "monkey", "penguin", "whale", "dolphin",
                    "car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane",
                    "van", "suv", "pickup truck", "ambulance", "police car", "fire truck",
                    "helicopter", "drone", "scooter", "kayak", "canoe", "yacht",
                    "chair", "couch", "bed", "dining table", "desk", "stool", "bench",
                    "wardrobe", "cabinet", "shelf", "bookshelf", "dresser", "nightstand",
                    "sofa", "recliner", "ottoman", "counter",
                    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
                    "plate", "pot", "pan", "kettle", "mug", "glass", "can", "jar",
                    "pizza", "cake", "sandwich", "banana", "apple", "orange", "broccoli", "carrot",
                    "bread", "cheese", "egg", "rice", "pasta", "steak", "sushi", "taco",
                    "donut", "cookie", "ice cream", "chocolate", "coffee", "tea",
                    "laptop", "cell phone", "keyboard", "mouse", "monitor", "tv",
                    "tablet", "headphones", "earbuds", "speaker", "camera", "tripod",
                    "remote control", "charger", "cable", "power outlet",
                    "book", "notebook", "pen", "pencil", "scissors", "stapler",
                    "printer", "calculator", "folder", "envelope", "tape",
                    "backpack", "handbag", "suitcase", "wallet", "briefcase",
                    "umbrella", "tie", "hat", "sunglasses", "glasses", "watch",
                    "ring", "necklace", "bracelet", "gloves", "scarf", "belt",
                    "clock", "vase", "lamp", "mirror", "picture frame", "curtain",
                    "rug", "carpet", "pillow", "blanket", "towel", "candle",
                    "chandelier", "fan", "air conditioner", "heater",
                    "tree", "flower", "grass", "mountain", "sky", "cloud",
                    "river", "lake", "ocean", "beach", "sand", "rock", "stone",
                    "sun", "moon", "star", "rainbow", "snow", "leaf",
                    "building", "house", "door", "window", "wall", "roof",
                    "tower", "bridge", "fence", "gate", "garage", "porch", "balcony",
                    "road", "street", "sidewalk", "crosswalk", "parking lot",
                    "traffic light", "stop sign", "street sign", "street lamp",
                    "football", "soccer ball", "basketball", "baseball bat", "tennis racket",
                    "golf club", "hockey stick", "ping pong paddle", "volleyball",
                    "helmet", "goggles", "ski", "snowboard", "surfboard", "skateboard",
                    "gym", "dumbbell", "yoga mat",
                    "guitar", "piano", "violin", "drum", "microphone", "keyboard",
                    "ukulele", "saxophone", "trumpet", "flute",
                    "teddy bear", "potted plant", "trash can", "fire hydrant", "mailbox",
                    "bench", "playground", "swing", "fountain",
                    "microwave", "oven", "toaster", "refrigerator", "blender", "coffee maker",
                    "dishwasher", "washing machine", "dryer",
                ],
                "confidence": 0.65,
            },
        )

        if resp.status_code == 200:
            data = resp.json()
            objects = _parse_yolo_world_objects(data)
            return {"objects": objects}

        logger.warning("Detect failed %d: %s", resp.status_code, resp.text[:500])
        return {"objects": [], "error": f"API {resp.status_code}"}

    except Exception as e:
        logger.error("detect exception: %s", e, exc_info=True)
        return {"objects": [], "error": str(e)}


def _parse_yolo_world_objects(data: any) -> list:
    """Parse YOLO-World Roboflow response into objects list with bboxes."""
    objects = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                obj = _make_object(item)
                if obj:
                    objects.append(obj)
        return objects

    if not isinstance(data, dict):
        return objects

    if "outputs" in data and isinstance(data["outputs"], list):
        for output in data["outputs"]:
            if isinstance(output, dict):
                preds = output.get("predictions", output)
                if isinstance(preds, dict):
                    pred_list = preds.get("predictions", preds.get("detections", []))
                    if isinstance(pred_list, list):
                        for p in pred_list:
                            obj = _make_object(p)
                            if obj:
                                objects.append(obj)
                elif isinstance(pred_list, list):
                    for p in pred_list:
                        obj = _make_object(p)
                        if obj:
                            objects.append(obj)
        if objects:
            return objects

    for key in ["predictions", "detections", "results", "data"]:
        if key in data and isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, dict):
                    obj = _make_object(item)
                    if obj:
                        objects.append(obj)
            if objects:
                return objects

    if "class" in data or "label" in data:
        obj = _make_object(data)
        if obj:
            objects.append(obj)

    return objects


def _make_object(det: dict):
    """Convert a YOLO-World detection dict to our format."""
    label = det.get("class") or det.get("label") or det.get("name") or det.get("text") or ""
    confidence = det.get("confidence") or det.get("score") or det.get("conf") or 0

    if not label:
        return None

    if isinstance(confidence, float) and confidence <= 1:
        confidence = round(confidence * 100)

    if "x" in det and "y" in det and ("width" in det or "w" in det):
        cx = float(det.get("x", 0))
        cy = float(det.get("y", 0))
        w = float(det.get("width", det.get("w", 100)))
        h = float(det.get("height", det.get("h", 100)))
        x = cx - w / 2
        y = cy - h / 2
    else:
        bbox = det.get("box") or det.get("bbox") or {}
        if isinstance(bbox, dict):
            x = bbox.get("x") or bbox.get("xmin") or bbox.get("left") or 0
            y = bbox.get("y") or bbox.get("ymin") or bbox.get("top") or 0
            w = bbox.get("w") or bbox.get("width") or (bbox.get("xmax", 0) - bbox.get("xmin", 0)) or 100
            h = bbox.get("h") or bbox.get("height") or (bbox.get("ymax", 0) - bbox.get("ymin", 0)) or 100
        else:
            x, y, w, h = 10, 10, 80, 80

    return {
        "label": str(label).strip(),
        "confidence": int(confidence),
        "bbox": {"x": float(x), "y": float(y), "w": float(w), "h": float(h)},
    }


@app.post("/api/caption")
async def caption_image(file: UploadFile = File(...)):
    """Generate caption using Florence-2 — processes image in-memory from request payload."""
    try:
        contents = await file.read()
        logger.info("CAPTION: %s (%d bytes)", file.filename, len(contents))

        resp = _roboflow_workflow(
            workspace="mds-workspace-l14ck",
            workflow="florence2-large-demo",
            image_bytes=contents,
        )

        if resp.status_code == 200:
            data = resp.json()
            caption = _extract_caption(data)

            if caption:
                words = caption.split()
                mid = len(words) // 2
                captions = [
                    caption,
                    " ".join(words[mid:]) + " " + " ".join(words[:mid]) if mid > 0 else caption,
                    "A scene showing " + caption.lower().lstrip(),
                ]
                return {"captions": captions}

        logger.warning("Caption failed %d: %s", resp.status_code, resp.text[:500])
        return {"captions": [], "error": f"API {resp.status_code}"}

    except Exception as e:
        logger.error("caption exception: %s", e, exc_info=True)
        return {"captions": [], "error": str(e)}


def _extract_caption(data: any) -> str:
    """Recursively extract caption text from Florence-2 response."""
    if isinstance(data, str):
        return data.strip()

    if isinstance(data, list) and len(data) > 0:
        return _extract_caption(data[0])

    if isinstance(data, dict):
        for key in ["caption", "text", "generated_text", "output", "result"]:
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    return val.strip()
                if isinstance(val, list) and len(val) > 0:
                    return _extract_caption(val)
                if isinstance(val, dict):
                    return _extract_caption(val)

        for key in ["outputs", "results", "predictions"]:
            if key in data:
                return _extract_caption(data[key])

    return ""
