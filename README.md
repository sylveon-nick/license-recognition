# 🚗 License Plate Recognition Studio

An intelligent Computer Vision and Optical Character Recognition (OCR) application that detects vehicles, localizes license plates, crops the plate region, and extracts license text in real time.

Built with **YOLOv8**, **EasyOCR**, and a modern **Streamlit** user interface.

---

## ✨ Features

- **Automated Vehicle Detection**: Employs YOLOv8n to identify cars, buses, trucks, and motorcycles in frame.
- **Accurate Plate Localization**: Custom-trained YOLOv8 model detects license plates and extracts tight bounding boxes.
- **High-Accuracy OCR**: EasyOCR engine cleans and transcribes plate alphanumeric characters.
- **Modern Studio UI**:
  - Clean responsive design with authentic license plate visual styling.
  - Two input modes: **📁 File Upload** and **📸 Camera Capture** (with on-demand camera access to prevent unwanted browser permission prompts).
  - Configurable detection sensitivity sliders for both vehicle and plate models.
  - One-click CSV export and high-resolution plate crop previews.
- **Fast Cached Inference**: Pre-cached PyTorch models via Streamlit for fast subsequent predictions.

---

## 🛠️ Project Structure

```text
├── app.py                            # Streamlit web application & inference pipeline
├── util.py                           # Helper functions & CSV exporter
├── requirements.txt                  # Python package dependencies
├── packages.txt                      # System-level dependencies (libgl1)
├── models/
│   ├── yolov8n.pt                   # Pretrained YOLOv8 COCO model
│   └── license_plate_detector.pt     # Fine-tuned YOLOv8 plate detector
├── imgs/                             # UI assets & sample background
└── licenses_plates_imgs_detected/   # Destination for saved cropped plates
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites

- **Python 3.10+** (Python 3.11 recommended)
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/sylveon-nick/license-recognition.git
cd license-recognition
```

### 3. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(Optional: Using `uv` for ultra-fast setup)*
```bash
uv venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for Linux users**: You may need OpenGL system libraries:
> ```bash
> sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
> ```

---

## 💻 Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

1. Select **📁 Upload Image** to pick a vehicle photo, or **📸 Camera Capture** to take a live photo.
2. Click **⚡ Run Detection**.
3. View the annotated vehicle detections, cropped license plate, recognized plate number, and download the results CSV.

---

## ⚙️ Configuration & Model Weights

- You can tune detection thresholds using the sidebar sliders:
  - **License Plate Threshold**: Adjust to balance precision vs. recall on small or angled plates.
  - **Vehicle Detection Threshold**: Minimum confidence required for car/vehicle classification.
- Models are located in the `models/` folder. Custom YOLO weights can be replaced directly with identical naming.

---

## 📄 License

Open-source under the [MIT License](LICENSE).
