<div align="center">

# Automatic Number Plate Recognition System

### Real-time license plate detection, alignment, and recognition for Rwanda vehicle registration

[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-computer%20vision-green.svg)](https://opencv.org/)
[![Tesseract OCR](https://img.shields.io/badge/tesseract-OCR%20engine-orange.svg)](https://github.com/tesseract-ocr/tesseract)

</div>

---

## Overview

This project implements a complete **Automatic Number Plate Recognition (ANPR)** system inspired by a three-stage approach: detecting the plate, correcting its orientation, and extracting the text.

The application uses a webcam to capture live video, identifies potential number plates using OpenCV, straightens the detected plate, reads characters using OCR, and verifies them against Rwanda's standard plate format.

To improve reliability, the system confirms results over multiple frames before storing them in a CSV file.

---

## System Architecture

The ANPR pipeline consists of six interconnected stages that process video frames in real-time:

### Stage 1: Plate Detection

Each frame from the camera is converted to grayscale, filtered, and processed to highlight edges. The system then searches for rectangular shapes that resemble number plates based on size and proportions.

### Stage 2: Plate Alignment

Once a candidate region is found, it is transformed into a clean, straightened image using a perspective correction. The output is normalized to a fixed size of **450 × 140 pixels**.

### Stage 3: Text Recognition (OCR)

The processed plate image is enhanced and passed to Tesseract OCR, which extracts characters using only uppercase letters and digits.

### Stage 4: Format Validation

The recognized text is cleaned and checked against the Rwanda plate structure:

```
AAA999A
```

### Stage 5: Multi-Frame Confirmation

To reduce false detections, the same plate must appear consistently across several frames before it is accepted as valid.

### Stage 6: Data Storage

Verified plate numbers are saved into a CSV file located in the `data` directory.

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

Clone the repository:

```bash
git clone https://github.com/humuraelvin/car-plate-extraction.git
```

Set up Python (recommended version: **3.11.9**):

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verify Tesseract installation:

```bash
tesseract --version
```

If it's not detected, provide the full path using:

```
--tesseract-cmd
```

---

## Usage

You can test each stage independently:

**Camera test**

```bash
python src/camera.py
```

**Detection only**

```bash
python src/detect.py
```

**Detection + alignment**

```bash
python src/align.py
```

**Detection + alignment + OCR**

```bash
python src/ocr.py
```

**Full pipeline with validation**

```bash
python src/validate.py
```

**Complete system (recommended)**

```bash
python src/main.py
```

---

## Configuration Options

```bash
python src/main.py --camera 0 --width 1280 --height 720
python src/main.py --image sample.jpg
python src/main.py --buffer-size 5 --min-confirmations 3 --cooldown 10
```

---

## Testing Guidelines

1. Request permission from the vehicle owner before testing.
2. Launch the full system:

   ```bash
   python src/main.py
   ```
3. Position the camera to clearly capture the plate.
4. Wait until the system confirms the plate.
5. Press **`s`** to save screenshots.
6. Repeat with different vehicles.
7. Upload screenshots and results to your repository.

---

## Plate Format Specification

The system expects plates in the format:

```
AAA999A
```

**Examples:**

```
RAH972U
RAB123A
```

To improve accuracy, common OCR mistakes are corrected automatically (e.g., O ↔ 0, S ↔ 5, B ↔ 8, I ↔ 1).

---

## Sample Output

After testing, saved images will appear in the `screenshots` folder:

```markdown
![Detection](screenshots/detection.png)
![Alignment](screenshots/alignment.png)
![OCR](screenshots/ocr.png)
```

---

## Data Export Format

Recognized plates are stored as:

```csv
Plate Number,Timestamp
RAH972U,2026-03-12 10:45:16
```
