import os
import io
import json
import base64
import asyncio
import time
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION - API KEYS & SETTINGS
# ==========================================

# 🔑 API Keys - Đặt API key của bạn ở đây
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC3I5eXUqe0WWuH_UqYoulI6n-nqQlvO5o")

# 📁 Model Paths
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")

# 🎯 Detection Thresholds
PRED_CONF = float(os.getenv("PRED_CONF", "0.35"))  # Confidence threshold (0.0-1.0)
PRED_IOU = float(os.getenv("PRED_IOU", "0.45"))    # IOU threshold for NMS

# 🖼️ Image Processing Settings
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))  # Max dimension (increased for better detection)
MIN_IMAGE_SIDE = int(os.getenv("MIN_IMAGE_SIDE", "100"))   # Min dimension (very permissive)

# 🔄 Retry Configuration for API Calls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# ⚡ Performance Optimization Flags
ENABLE_SAFETY_CHECK = os.getenv("ENABLE_SAFETY_CHECK", "true").lower() == "true"
USE_PARALLEL_EXECUTION = os.getenv("USE_PARALLEL_EXECUTION", "true").lower() == "true"

# ==========================================
# 2. LOGGING SETUP
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 3. INITIALIZE AI MODELS
# ==========================================

# Setup Gemini AI
GEMINI_MODEL = None
GEMINI_AVAILABLE = False  # Track if Gemini is working

if GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
        GEMINI_AVAILABLE = True
        logger.info(f"✅ Gemini model loaded: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.error(f"❌ Gemini init error: {e}")
        GEMINI_MODEL = None
        GEMINI_AVAILABLE = False
else:
    logger.warning("⚠️ GEMINI_API_KEY not set! Gemini features will be disabled.")

# ==========================================
# 4. LABELS FOR YOUR TRAINED MODEL (11 classes)
# ==========================================

VALID_LABELS: List[str] = [
    "aluminum_can",
    "aluminum_caps",
    "cardboard",
    "combined_plastic",
    "foil",
    "milk_bottle",
    "paper_bag",
    "paper_cups",
    "plastic_bag",
    "plastic_bottle",
    "plastic_cup",
]

LABEL_MAP_VI: Dict[str, str] = {
    "plastic_bottle": "Chai nhựa",
    "aluminum_can": "Lon nhôm",
    "cardboard": "Bìa carton",
    "plastic_bag": "Túi nilon",
    "aluminum_caps": "Nắp chai kim loại",
    "plastic_cup": "Cốc nhựa",
    "paper_cups": "Cốc giấy",
    "paper_bag": "Túi giấy",
    "milk_bottle": "Hộp sữa/nước",
    "combined_plastic": "Bao bì nhựa",
    "foil": "Giấy bạc"
}

YOLO_NAME_TO_KEY: Dict[str, str] = {
    "Aluminum can": "aluminum_can",
    "Aluminum caps": "aluminum_caps",
    "Cardboard": "cardboard",
    "Combined plastic": "combined_plastic",
    "Foil": "foil",
    "Milk bottle": "milk_bottle",
    "Paper bag": "paper_bag",
    "Paper cups": "paper_cups",
    "Plastic bag": "plastic_bag",
    "Plastic bottle": "plastic_bottle",
    "Plastic cup": "plastic_cup",
}

# ==========================================
# 5. FASTAPI INIT
# ==========================================

app = FastAPI(
    title="AI 4 Green API",
    description="Recyclable waste detection API using YOLO + Gemini",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model with better error handling
model = None
try:
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        logger.info(f"✅ YOLO model loaded from: {MODEL_PATH}")
        logger.info(f"📊 Model classes: {model.names}")
except Exception as e:
    logger.error(f"❌ YOLO load error: {e}")
    model = None

# ==========================================
# 6. UTILITY FUNCTIONS
# ==========================================

def optimize_image(content: bytes) -> Tuple[Image.Image, Optional[str]]:
    """
    Optimized image preprocessing: validation + resizing in single pass
    Returns (image, error_message)
    """
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        width, height = img.size
        
        # Validate dimensions - at least one side should meet minimum
        if width < MIN_IMAGE_SIDE and height < MIN_IMAGE_SIDE:
            return None, f"Image too small. At least one dimension must be >= {MIN_IMAGE_SIDE}px"
        
        if width > MAX_IMAGE_SIDE * 2 or height > MAX_IMAGE_SIDE * 2:
            return None, f"Image too large. Maximum size: {MAX_IMAGE_SIDE*2}x{MAX_IMAGE_SIDE*2}px"
        
        # Resize if needed
        if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        
        return img, None
        
    except Exception as e:
        return None, f"Invalid image file: {str(e)}"

def validate_image(img: Image.Image) -> Tuple[bool, str]:
    """Validate image dimensions and format"""
    width, height = img.size
    
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        return False, f"Image too small. Minimum size: {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE}px"
    
    if width > MAX_IMAGE_SIDE * 2 or height > MAX_IMAGE_SIDE * 2:
        return False, f"Image too large. Maximum size: {MAX_IMAGE_SIDE*2}x{MAX_IMAGE_SIDE*2}px"
    
    return True, ""

def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def normalize_key(text: str) -> str:
    """Normalize label text to canonical key format"""
    text = text.lower().strip()
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text

def fuzzy_match_label(raw_label: str) -> Optional[str]:
    """Match raw label to canonical key with fuzzy matching"""
    norm = normalize_key(raw_label)

    # Direct match on canonical keys
    if norm in VALID_LABELS:
        return norm

    # Match YOLO class names
    title = raw_label.strip()
    if title in YOLO_NAME_TO_KEY:
        return YOLO_NAME_TO_KEY[title]

    # Substring matching
    for k in VALID_LABELS:
        if k in norm or norm in k:
            return k

    return None

def robust_json_parse(text: str) -> Optional[Dict | List]:
    """Parse JSON from various formats including markdown code blocks"""
    if not text:
        return None
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code blocks
    patterns = [
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        r"(\{.*?\}|\[.*?\])",
    ]
    
    for pattern in patterns:
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            continue
    
    return None

# ==========================================
# 7. GEMINI FUNCTIONS WITH RETRY
# ==========================================

async def call_gemini_with_retry(prompt: str, img: Image.Image, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """Call Gemini API with exponential backoff retry"""
    global GEMINI_AVAILABLE
    
    if not GEMINI_MODEL or not GEMINI_AVAILABLE:
        return None
    
    for attempt in range(max_retries):
        try:
            # Use asyncio.to_thread for CPU-bound operation
            response = await asyncio.to_thread(
                GEMINI_MODEL.generate_content,
                [prompt, img]
            )
            
            if response and response.text:
                return response.text
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for quota/rate limit errors
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                logger.error(f"❌ Gemini quota exceeded! Disabling Gemini for this session.")
                GEMINI_AVAILABLE = False  # Disable globally
                return None
            
            logger.warning(f"Gemini API attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(delay)
            else:
                logger.error(f"Gemini API failed after {max_retries} attempts")
                # Disable Gemini after multiple failures
                GEMINI_AVAILABLE = False
    
    return None

async def check_safety(img: Image.Image) -> Tuple[bool, str]:
    """Check image for dangerous/inappropriate content"""
    if not GEMINI_MODEL or not GEMINI_AVAILABLE:
        logger.info("Safety check skipped (Gemini not available)")
        return True, "Skipped"
    
    try:
        prompt = (
            "Analyze this image for dangerous or inappropriate items such as weapons, fire, "
            "toxic materials, drugs, or explicit content. "
            'Return ONLY a JSON object: {"is_safe": true/false, "reason": "explanation"}. '
            "If the image only contains recyclable waste, return is_safe: true."
        )
        
        response_text = await call_gemini_with_retry(prompt, img)
        
        if not response_text:
            return True, "Check failed"
        
        parsed = robust_json_parse(response_text)
        
        if isinstance(parsed, dict):
            is_safe = bool(parsed.get("is_safe", True))
            reason = str(parsed.get("reason", ""))
            logger.info(f"Safety check: {'✅ Safe' if is_safe else '⚠️ Unsafe'} - {reason}")
            return is_safe, reason
        
        return True, "Invalid response format"
        
    except Exception as e:
        logger.error(f"Safety check error: {e}")
        return True, "Error during check"

async def scan_gemini_labels(img: Image.Image) -> List[Dict]:
    """
    Scan image for recyclable items using Gemini vision
    Returns list of dicts with 'key' and 'name_vi' for each detected item
    """
    if not GEMINI_MODEL or not GEMINI_AVAILABLE:
        if not GEMINI_AVAILABLE:
            logger.warning("⚠️ Gemini disabled - using YOLO-only mode")
        return []
    
    try:
        allowed_str = ", ".join(VALID_LABELS)
        prompt = (
            f"Phân tích ảnh này và xác định TẤT CẢ các vật liệu có thể tái chế.\n\n"
            f"**Ưu tiên các vật liệu này (nếu có):** {allowed_str}\n\n"
            f"**Quan trọng:** Nếu phát hiện vật liệu tái chế KHÁC không nằm trong danh sách trên "
            f"(ví dụ: chai thủy tinh, kim loại, vải, giấy báo, hộp carton sữa, v.v.), "
            f"hãy VẪN BÁO CÁO chúng.\n\n"
            f"Trả về JSON array với format:\n"
            f"[\n"
            f'  {{"key": "plastic_bottle", "name_vi": "Chai nhựa"}},\n'
            f'  {{"key": "glass_bottle", "name_vi": "Chai thủy tinh"}},\n'
            f'  {{"key": "metal_can", "name_vi": "Lon kim loại"}}\n'
            f"]\n\n"
            f"- 'key': tên tiếng Anh viết thường, dùng dấu gạch dưới (ví dụ: glass_bottle)\n"
            f"- 'name_vi': tên tiếng Việt tự nhiên, dễ hiểu\n"
            f"- Nếu không có vật liệu tái chế nào, trả về: []\n"
            f"- KHÔNG thêm text giải thích, CHỈ trả về JSON array"
        )
        
        response_text = await call_gemini_with_retry(prompt, img)
        
        if not response_text:
            logger.warning("Gemini scan returned no response")
            return []
        
        parsed = robust_json_parse(response_text)
        
        if not isinstance(parsed, list):
            logger.warning(f"Gemini returned non-list: {type(parsed)}")
            return []
        
        valid_results: List[Dict] = []
        for item in parsed:
            if isinstance(item, dict) and "key" in item and "name_vi" in item:
                key = str(item["key"]).strip()
                name_vi = str(item["name_vi"]).strip()
                
                # Normalize key format
                key = normalize_key(key)
                
                # Try to match with known labels
                matched = fuzzy_match_label(key)
                
                if matched:
                    # Known label - use predefined Vietnamese name
                    valid_results.append({
                        "key": matched,
                        "name_vi": LABEL_MAP_VI.get(matched, name_vi),
                        "is_known": True
                    })
                    logger.info(f"🧠 Gemini detected (known): {matched} - {LABEL_MAP_VI.get(matched)}")
                else:
                    # Unknown label - use Gemini's Vietnamese name
                    valid_results.append({
                        "key": key,
                        "name_vi": name_vi,
                        "is_known": False
                    })
                    logger.info(f"🧠 Gemini detected (unknown): {key} - {name_vi}")
            else:
                logger.warning(f"Gemini returned invalid item format: {item}")
        
        logger.info(f"🧠 Gemini total detected: {len(valid_results)} items")
        return valid_results
        
    except Exception as e:
        logger.error(f"Gemini scan error: {e}")
        return []

# ==========================================
# 8. DETECTION LOGIC
# ==========================================

def run_yolo_detection(img: Image.Image) -> Tuple[Counter, List[Dict]]:
    """Run YOLO object detection on image (optimized for parallel execution)"""
    yolo_counts = Counter()
    detections = []
    
    if not model:
        logger.warning("YOLO model not available")
        return yolo_counts, detections
    
    try:
        # Use half precision for faster inference (if GPU available)
        results = model.predict(
            img,
            conf=PRED_CONF,
            iou=PRED_IOU,
            agnostic_nms=True,
            verbose=False,
            max_det=100,
            half=False,  # Set to True if using GPU
        )
        
        if not results:
            return yolo_counts, detections
        
        r = results[0]
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            yolo_name = r.names.get(cls_id, "unknown")
            key = YOLO_NAME_TO_KEY.get(yolo_name, "unknown")
            conf = float(box.conf[0])
            
            yolo_counts[key] += 1
            detections.append({
                "box": box.xyxy[0].tolist(),
                "key": key,
                "label_vi": LABEL_MAP_VI.get(key, key),
                "conf": round(conf, 3),
            })
        
        if yolo_counts:
            logger.info(f"👁️ YOLO detected: {dict(yolo_counts)}")
        
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
    
    return yolo_counts, detections

def merge_detections(yolo_counts: Counter, gemini_items: List[Dict]) -> List[Dict]:
    """
    Optimized merge algorithm:
    - If Gemini found items: use Gemini for classification, YOLO for counting
    - If Gemini failed: fallback to YOLO only
    - Handle both known labels (11 YOLO classes) and unknown materials
    - Mark items needing manual verification
    """
    final_items = []
    
    if not gemini_items:
        # Fallback: YOLO only
        logger.info("Using YOLO-only mode (Gemini unavailable)")
        for key, count in yolo_counts.items():
            if key != "unknown":
                final_items.append({
                    "name": key,
                    "label": LABEL_MAP_VI.get(key, key),
                    "quantity": int(count),
                    "manual_input_required": False,
                    "note": "YOLO detection",
                })
        return final_items
    
    # Hybrid mode: Gemini + YOLO
    processed = set()
    
    # Pass 1: Process all items detected by Gemini
    for item in gemini_items:
        key = item["key"]
        name_vi = item["name_vi"]
        is_known = item.get("is_known", False)
        
        processed.add(key)
        yolo_qty = int(yolo_counts.get(key, 0))
        
        if is_known and yolo_qty > 0:
            # Best case: Known material + YOLO detected it
            final_items.append({
                "name": key,
                "label": name_vi,
                "quantity": yolo_qty,
                "manual_input_required": False,
                "note": "Verified by AI",
            })
        elif is_known and yolo_qty == 0:
            # Known material but YOLO missed it
            final_items.append({
                "name": key,
                "label": name_vi,
                "quantity": 0,
                "manual_input_required": True,
                "note": "Please count manually",
            })
        else:
            # Unknown material - YOLO cannot detect it
            final_items.append({
                "name": key,
                "label": name_vi,
                "quantity": 0,
                "manual_input_required": True,
                "note": "Detected by Gemini - Please count manually",
            })
    
    # Pass 2: Items YOLO found but Gemini missed
    for key, count in yolo_counts.items():
        if key not in processed and key != "unknown":
            final_items.append({
                "name": key,
                "label": LABEL_MAP_VI.get(key, key),
                "quantity": int(count),
                "manual_input_required": True,
                "note": "Please verify",
            })
    
    logger.info(f"📦 Final items: {len(final_items)}")
    return final_items

def draw_detections(img: Image.Image, detections: List[Dict]) -> Image.Image:
    """Draw bounding boxes on image"""
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    for det in detections:
        box = det["box"]
        conf = det["conf"]
        label = det["label_vi"]
        
        # Draw box
        draw.rectangle(box, outline="lime", width=3)
        
        # Draw label background
        text = f"{label} ({conf:.2f})"
        try:
            # Try to use default font
            bbox = draw.textbbox((box[0], box[1] - 20), text)
            draw.rectangle(bbox, fill="lime")
            draw.text((box[0], box[1] - 20), text, fill="black")
        except:
            # Fallback if font issues
            pass
    
    return draw_img

# ==========================================
# 9. API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "2.0.0",
        "models": {
            "yolo": "loaded" if model else "unavailable",
            "gemini": "loaded" if GEMINI_MODEL else "unavailable"
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint (OPTIMIZED)
    - Accepts image file
    - Returns detected recyclable items with quantities
    - Flow: Safety check → Parallel detection (YOLO + Gemini)
    """
    start_time = time.time()
    logger.info(f"\n{'='*50}")
    logger.info(f"📸 Processing: {file.filename}")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and optimize image (combined validation + resizing)
    try:
        content = await file.read()
        img, error_msg = optimize_image(content)
        
        if error_msg:
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"📐 Image size: {img.size}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Step 1: Safety check FIRST (if enabled)
    if ENABLE_SAFETY_CHECK:
        is_safe, safety_reason = await check_safety(img)
        if not is_safe:
            logger.warning(f"⚠️ Safety check failed: {safety_reason}")
            latency = time.time() - start_time
            logger.info(f"❌ Blocked in {latency:.2f}s")
            logger.info(f"{'='*50}\n")
            return JSONResponse(
                {
                    "items": [],
                    "error": "SAFETY_BLOCKED",
                    "message": f"Image blocked: {safety_reason}",
                    "detections": [],
                    "latency_s": round(latency, 3),
                },
                status_code=400,
            )
    
    # Step 2: Run YOLO + Gemini in parallel (only if safe)
    if USE_PARALLEL_EXECUTION:
        try:
            # Execute YOLO and Gemini concurrently
            (yolo_counts, detections), gemini_items = await asyncio.gather(
                asyncio.to_thread(run_yolo_detection, img),  # YOLO in thread
                scan_gemini_labels(img),                      # Gemini async
            )
        except Exception as e:
            logger.error(f"Parallel execution error: {e}")
            raise HTTPException(status_code=500, detail="Detection failed")
    else:
        # Sequential execution (fallback)
        yolo_counts, detections = run_yolo_detection(img)
        gemini_items = await scan_gemini_labels(img)
    
    # Step 3: Merge results
    final_items = merge_detections(yolo_counts, gemini_items)
    
    # Step 4: Draw boxes
    annotated_img = draw_detections(img, detections)
    
    # Calculate latency
    latency = time.time() - start_time
    logger.info(f"✅ Completed in {latency:.2f}s")
    logger.info(f"{'='*50}\n")
    
    return {
        "items": final_items,
        "detections": detections,
        "image": pil_to_base64(annotated_img),
        "latency_s": round(latency, 3),
        "metadata": {
            "yolo_count": sum(yolo_counts.values()),
            "gemini_items": len(gemini_items),
            "final_items": len(final_items),
            "parallel_execution": USE_PARALLEL_EXECUTION,
            "safety_check_enabled": ENABLE_SAFETY_CHECK,
            "gemini_available": GEMINI_AVAILABLE,
        }
    }

# ==========================================
# 10. MAIN
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting AI 4 Green API...")
    logger.info(f"📊 Configuration:")
    logger.info(f"  - YOLO Model: {MODEL_PATH}")
    logger.info(f"  - Gemini Model: {GEMINI_MODEL_NAME}")
    logger.info(f"  - Confidence: {PRED_CONF}")
    logger.info(f"  - IOU: {PRED_IOU}")
    logger.info(f"⚡ Performance:")
    logger.info(f"  - Parallel Execution: {USE_PARALLEL_EXECUTION}")
    logger.info(f"  - Safety Check: {ENABLE_SAFETY_CHECK}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
