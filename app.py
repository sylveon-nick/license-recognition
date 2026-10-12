import os
import uuid
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import easyocr

from util import save_detection_csv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="License Plate Recognition Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN POLISHED STYLES ---
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    
    /* Header styling */
    .header-badge {
        display: inline-block;
        padding: 5px 14px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        color: #60a5fa;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Authentic License Plate Design */
    .plate-badge-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0.75rem 0;
        width: 100%;
    }
    
    .license-plate-ui {
        background: #ffffff;
        border: 2.5px solid #0f172a;
        border-radius: 7px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        white-space: nowrap;
        max-width: 100%;
        box-sizing: border-box;
    }
    
    .plate-strip {
        background: #1d4ed8;
        color: #fbbf24;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 4px 7px;
        border-radius: 3px;
        margin-right: 14px;
        text-align: center;
        line-height: 1.2;
        letter-spacing: 0.5px;
        flex-shrink: 0;
    }
    
    .plate-number-text {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: 2.5px;
        color: #0f172a;
        text-transform: uppercase;
        white-space: nowrap;
        line-height: 1.2;
    }
    
    /* Empty State */
    .empty-hero {
        text-align: center;
        padding: 3rem 1.5rem;
        border: 2px dashed rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.3);
    }
    .empty-hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .empty-hero-text {
        color: #94a3b8;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- PATH CONSTANTS ---
CROPS_DIR = "./licenses_plates_imgs_detected"
CSV_PATH = "./csv_detections/detection_results.csv"
COCO_MODEL_DIR = "./models/yolov8n.pt"
LICENSE_MODEL_DIR = "./models/license_plate_detector.pt"

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# COCO vehicle classes: car (2), motorcycle (3), bus (5), truck (7)
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- CACHED MODEL LOADER ---
@st.cache_resource(show_spinner="Loading detection models into memory...")
def get_models():
    coco = YOLO(COCO_MODEL_DIR)
    license_detector = YOLO(LICENSE_MODEL_DIR)
    reader = easyocr.Reader(['en'], gpu=False)
    return coco, license_detector, reader

coco_model, license_detector, reader = get_models()

# --- OCR PROCESSING HELPER ---
def extract_plate_text(plate_crop, min_confidence=0.1):
    detections = reader.readtext(plate_crop)
    if not detections:
        return None, 0.0

    rect_area = plate_crop.shape[0] * plate_crop.shape[1]
    filtered_tokens = []
    scores = []

    for item in detections:
        bbox, text, score = item
        if score < min_confidence:
            continue
        
        # Calculate token area ratio relative to crop
        box_width = np.sum(np.subtract(bbox[1], bbox[0]))
        box_height = np.sum(np.subtract(bbox[2], bbox[1]))
        if (box_width * box_height) / max(rect_area, 1) > 0.08:
            clean_text = text.strip().upper()
            if clean_text:
                filtered_tokens.append(clean_text)
                scores.append(score)

    if filtered_tokens:
        full_text = " ".join(filtered_tokens)
        avg_score = float(np.mean(scores))
        return full_text, avg_score
    
    # Fallback to highest confidence text if area filter was too strict
    best_item = max(detections, key=lambda x: x[2])
    return best_item[1].strip().upper(), float(best_item[2])

# --- INFERENCE ENGINE ---
def run_pipeline(img_rgb, vehicle_conf=0.25, plate_conf=0.25):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = img_bgr.copy()

    # 1. Detect Vehicles
    vehicle_boxes = []
    vehicle_preds = coco_model(img_bgr, conf=vehicle_conf)[0]
    if len(vehicle_preds.boxes) > 0:
        for det in vehicle_preds.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if int(class_id) in VEHICLE_CLASSES:
                vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2), score))
                # Draw vehicle box
                cv2.rectangle(annotated_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (230, 160, 50), 2)
                label = f"Vehicle {score*100:.0f}%"
                cv2.putText(annotated_bgr, label, (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 160, 50), 2)

    # 2. Detect License Plates
    plate_preds = license_detector(img_bgr, conf=plate_conf)[0]
    detections_data = []

    if len(plate_preds.boxes) > 0:
        for idx, plate_box in enumerate(plate_preds.boxes.data.tolist()):
            x1, y1, x2, y2, p_score, _ = plate_box
            ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(img_bgr.shape[1], int(x2)), min(img_bgr.shape[0], int(y2))
            
            crop_bgr = img_bgr[iy1:iy2, ix1:ix2]
            if crop_bgr.size == 0:
                continue

            # Save crop to disk
            crop_filename = f"{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(CROPS_DIR, crop_filename)
            cv2.imwrite(crop_path, crop_bgr)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # OCR Extraction
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            plate_text, ocr_score = extract_plate_text(crop_gray)

            # Match plate to surrounding vehicle if any
            matched_car_box = [0, 0, 0, 0]
            matched_car_score = 0.0
            for vx1, vy1, vx2, vy2, vscore in vehicle_boxes:
                if vx1 <= ix1 and vy1 <= iy1 and vx2 >= ix2 and vy2 >= iy2:
                    matched_car_box = [vx1, vy1, vx2, vy2]
                    matched_car_score = vscore
                    break

            # Draw Plate Bounding Box (vibrant emerald green)
            cv2.rectangle(annotated_bgr, (ix1, iy1), (ix2, iy2), (0, 220, 115), 3)
            plate_tag = f"{plate_text or 'Plate'} ({p_score*100:.0f}%)"
            
            # Badge background for text
            (tw, th), _ = cv2.getTextSize(plate_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated_bgr, (ix1, max(0, iy1 - th - 10)), (ix1 + tw + 10, iy1), (0, 220, 115), -1)
            cv2.putText(annotated_bgr, plate_tag, (ix1 + 5, max(th, iy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            detections_data.append({
                'car_id': idx + 1,
                'plate_text': plate_text or "UNREADABLE",
                'text_score': ocr_score,
                'plate_score': p_score,
                'car_score': matched_car_score,
                'plate_bbox': [ix1, iy1, ix2, iy2],
                'car_bbox': matched_car_box,
                'crop_image': crop_rgb,
                'crop_filename': crop_filename
            })

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections_data

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.caption("Fine-tune model sensitivity and thresholds")
    
    plate_threshold = st.slider("License Plate Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05,
                                help="Minimum confidence required to detect a license plate.")
    vehicle_threshold = st.slider("Vehicle Detection Threshold", min_value=0.10, max_value=0.90, value=0.30, step=0.05,
                                 help="Minimum confidence required to identify surrounding vehicles.")
    
    st.divider()
    st.markdown("### 🧠 Loaded Models")
    st.markdown("""
    - **Plate Detector**: YOLOv8 Custom (`license_plate_detector.pt`)
    - **Vehicle Detector**: YOLOv8n COCO (`yolov8n.pt`)
    - **Text Extraction**: EasyOCR v1.7
    """)
    
    st.divider()
    st.caption("Cleaned & Optimized Edition")

# --- HEADER SECTION ---
st.markdown('<div class="header-badge">Computer Vision & OCR</div>', unsafe_allow_html=True)
st.markdown('<h1 class="header-title">License Plate Recognition Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Upload a vehicle photo or capture an image to identify vehicle boundaries, crop plates, and read license numbers instantly.</p>', unsafe_allow_html=True)

# --- INPUT SELECTOR TABS ---
# --- INPUT SELECTOR ---
input_mode = st.segmented_control(
    "Input Mode",
    options=["📁 Upload Image", "📸 Camera Capture"],
    default="📁 Upload Image",
    label_visibility="collapsed"
)
if input_mode is None:
    input_mode = "📁 Upload Image"

input_image = None
image_key = None

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
else:
    camera_photo = st.camera_input("Take a photo of a vehicle", label_visibility="collapsed")
    if camera_photo is not None:
        input_image = Image.open(camera_photo).convert("RGB")
        image_key = f"camera_{camera_photo.size}"

# Clear cached detection if image changed
if "current_image_key" not in st.session_state or st.session_state["current_image_key"] != image_key:
    st.session_state["current_image_key"] = image_key
    st.session_state["detection_results"] = None

# --- MAIN CONTENT AREA ---
if input_image is None:
    st.markdown("""
    <div class="empty-hero">
        <div class="empty-hero-icon">🚗</div>
        <div style="font-size: 1.25rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.25rem;">
            No Image Selected
        </div>
        <div class="empty-hero-text">
            Upload a vehicle photo or use your camera above to begin detection.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    img_np = np.array(input_image)

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.markdown("### 📷 Input Image")
        st.image(input_image, use_container_width=True)
        
        run_btn = st.button("⚡ Run Detection", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Processing image with YOLOv8 and EasyOCR..."):
            annotated_img, detections = run_pipeline(
                img_np,
                vehicle_conf=vehicle_threshold,
                plate_conf=plate_threshold
            )
            st.session_state["detection_results"] = {
                "annotated_img": annotated_img,
                "detections": detections
            }

    # Render results if available in session state
    if st.session_state.get("detection_results") is not None:
        cached_res = st.session_state["detection_results"]
        annotated_img = cached_res["annotated_img"]
        detections = cached_res["detections"]

        with col_right:
            st.markdown("### 🎯 Detection Results")
            st.image(annotated_img, use_container_width=True)

            if not detections:
                st.warning("No license plates detected. Try lowering the threshold slider in the sidebar.")
            else:
                st.markdown("---")
                st.markdown(f"#### 🔎 Identified Plates ({len(detections)})")

                for det in detections:
                    with st.container(border=True):
                        plate_text = det['plate_text']
                        
                        # Authentic Single-Line License Plate Badge
                        st.markdown(f"""
                        <div class="plate-badge-wrapper">
                            <div class="license-plate-ui">
                                <div class="plate-strip">IND</div>
                                <div class="plate-number-text">{plate_text}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Cropped Plate Image
                        st.caption("Cropped License Plate")
                        st.image(det['crop_image'], use_container_width=True)
                        
                        # Metrics in 3 clean columns
                        m1, m2, m3 = st.columns(3)
                        m1.metric("OCR Conf", f"{det['text_score']*100:.1f}%")
                        m2.metric("Plate Conf", f"{det['plate_score']*100:.1f}%")
                        if det['car_score'] > 0:
                            m3.metric("Vehicle Conf", f"{det['car_score']*100:.1f}%")
                        else:
                            m3.metric("Vehicle Conf", "N/A")

                # Save and render clean CSV table
                df_results = save_detection_csv(detections, CSV_PATH)
                
                st.markdown("#### 📊 Detection Data")
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                csv_bytes = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_bytes,
                    file_name="license_plate_detections.csv",
                    mime="text/csv",
                    use_container_width=True
                )