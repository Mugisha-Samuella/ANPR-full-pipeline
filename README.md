

# Automatic Number Plate Recognition System

### Real-Time License Plate Detection, Alignment, and Recognition for Rwanda Vehicle Registration


---

## Overview

This project implements a comprehensive **Automatic Number Plate Recognition (ANPR)** system, following a three-stage architecture: plate detection, orientation correction, and optical character recognition (OCR). The application leverages a webcam for live video capture, employs OpenCV for candidate plate identification, corrects perspective distortion, extracts text via Tesseract OCR, and validates the output against the official Rwanda vehicle registration format. To ensure robustness, the system employs temporal verification across multiple frames before persisting validated results to a structured CSV file.

---

## System Architecture

The ANPR pipeline is composed of six integrated processing stages, operating sequentially on each video frame:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Plate Detection | Converts each frame to grayscale, applies morphological filtering and edge detection, then identifies rectangular contours matching the geometric and dimensional characteristics of a standard license plate. |
| 2 | Plate Alignment | Applies perspective transformation to extract the detected region, corrects for skew and rotation, and normalizes the output to a fixed resolution of **450 × 140 pixels**. |
| 3 | Text Recognition (OCR) | Enhances the normalized plate image through binarization and noise reduction, then extracts alphanumeric characters using Tesseract OCR configured for uppercase letters and digits. |
| 4 | Format Validation | Sanitizes the OCR output and validates the string against the Rwanda plate structure pattern: `AAA999A`. |
| 5 | Multi-Frame Confirmation | Implements temporal consistency by requiring a candidate plate to be recognized successfully across a configurable number of consecutive frames before acceptance. |
| 6 | Data Storage | Appends verified plate numbers along with a timestamp to a CSV file located in the designated `data` directory. |

---

## Project Structure

```
car_plate_extraction/
├── README.md
├── requirements.txt
├── data/
│   └── plates.csv
├── screenshots/
└── src/
    ├── camera.py
    ├── detect.py
    ├── align.py
    ├── ocr.py
    ├── validate.py
    ├── main.py
    └── pipeline.py
```

---

## Installation

Follow these steps to set up the project environment:

1. Clone the repository:

```bash
git clone https://github.com/Mugisha-Samuella/ANPR-full-pipeline.git
cd ANPR-full-pipeline
```

2. Set up Python environment (version **3.11.9** recommended):

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Verify Tesseract OCR installation:

```bash
tesseract --version
```

If Tesseract is not in your system PATH, specify its full path using the `--tesseract-cmd` argument when running the application.

---

## Usage

The system provides modular components for testing and a unified entry point for full operation.

### Component Testing

| Command | Description |
|---------|-------------|
| `python src/camera.py` | Tests camera feed and basic capture functionality. |
| `python src/detect.py` | Executes only the plate detection stage with visual feedback. |
| `python src/align.py` | Performs detection and perspective correction. |
| `python src/ocr.py` | Runs detection, alignment, and OCR extraction. |
| `python src/validate.py` | Executes the complete pipeline including format validation. |

### Full System Execution

Launch the complete ANPR system with default settings:

```bash
python src/main.py
```

---

## Configuration Options

The main application supports several command-line arguments for customization:

```bash
# Specify camera device and resolution
python src/main.py --camera 0 --width 1280 --height 720

# Process a single image file instead of camera feed
python src/main.py --image sample.jpg

# Adjust temporal validation parameters
python src/main.py --buffer-size 5 --min-confirmations 3 --cooldown 10
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--camera` | Camera device index | 0 |
| `--width` | Capture frame width | 1280 |
| `--height` | Capture frame height | 720 |
| `--image` | Path to image file (overrides camera) | None |
| `--buffer-size` | Frames to maintain in history | 5 |
| `--min-confirmations` | Required consistent detections | 3 |
| `--cooldown` | Seconds to wait after successful detection | 10 |

---

## Testing Guidelines

To ensure accurate evaluation and reproducible results:

1. **Obtain consent** from vehicle owners before conducting tests.
2. Launch the system in full mode:

```bash
python src/main.py
```

3. Position the camera such that the license plate occupies a significant portion of the frame and is well-lit.
4. Maintain a stable position until the system confirms detection (indicated on-screen).
5. Press **`s`** during runtime to capture and save screenshots of the current detection state.
6. Repeat the process with multiple vehicles to evaluate generalization.
7. Upload collected screenshots and the generated CSV data to the repository for documentation.

---

## Plate Format Specification

The system validates license plates against the official Rwanda format:

**Pattern:** `AAA999A`

- Positions 1–3: Uppercase letters (A–Z)
- Positions 4–6: Numeric digits (0–9)
- Position 7: Uppercase letter (A–Z)

**Valid Examples:**

```
RAH972U
RAB123A
```

**Character Correction:** The OCR post-processor automatically corrects common misreadings such as:

- `O` ↔ `0`
- `S` ↔ `5`
- `B` ↔ `8`
- `I` ↔ `1`

---

## Sample Output

Upon successful detection, the system saves annotated images to the `screenshots` directory, illustrating each stage of the pipeline:

| Stage | Example |
|-------|---------|
| Detection | ![Detection](screenshots/detection.png) |
| Alignment | ![Alignment](screenshots/alignment.png) |
| OCR Result | ![OCR](screenshots/ocr.png) |

---

## Data Export Format

All confirmed plate recognitions are logged in `data/plates.csv` with the following structure:

```csv
Plate Number,Timestamp
RAH972U,2026-03-12 10:45:16
RAB123A,2026-03-12 10:47:22
```

The timestamp records the moment of final confirmation after the multi-frame validation.
