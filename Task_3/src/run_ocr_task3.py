
#!/usr/bin/env python3
"""
Task 3 - OCR with structured extraction

This script:
1. Reads equipment/nameplate label images.
2. Runs OCR without preprocessing.
3. Applies OpenCV preprocessing and runs OCR again.
4. Extracts structured fields:
   - equipment_id
   - serial_number
   - model
   - voltage
   - manufacture_date
5. Saves:
   - preprocessed images
   - labels_output.json
   - labels_output_raw.json
   - accuracy_report.json
   - accuracy_report.md
6. Compares predictions against labels_ground_truth.csv and reports:
   - field-level accuracy
   - serial_number Character Error Rate, CER

Recommended command:

python task3/src/run_ocr_task3.py \
  --images_dir task3/data/images \
  --ground_truth task3/labels_ground_truth.csv \
  --out_dir task3/outputs \
  --engine easyocr

Optional Tesseract command:

python task3/src/run_ocr_task3.py \
  --images_dir task3/data/images \
  --ground_truth task3/labels_ground_truth.csv \
  --out_dir task3/outputs \
  --engine tesseract

CSV format:

image,equipment_id,serial_number,model,voltage,manufacture_date
img001.jpg,,A782F740,3196,,
img002.jpg,,258C051-1,3160,,

Use blank values for unavailable fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
# import zipfile
# from urllib.error import ContentTooShortError, URLError


FIELDS = [
    "equipment_id",
    "serial_number",
    "model",
    "voltage",
    "manufacture_date",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

NULL_STRINGS = {
    "",
    "NULL",
    "NONE",
    "N/A",
    "NA",
    "NAN",
    "-",
    "--",
}


FIELD_PATTERNS = {
    "equipment_id": [
        r"\bEQUIP(?:MENT)?\s*(?:ID|NO|NUMBER)?\b",
        r"\bASSET\s*(?:ID|NO|NUMBER)?\b",
        r"\bTAG\s*(?:ID|NO|NUMBER)?\b",
        r"\bCAT(?:ALOG)?\b",
        r"\bPART\s*(?:NO|NUMBER|#)?\b",
        r"\bP\s*/?\s*N\b",
    ],
    "serial_number": [
        r"\bSERIAL\s*(?:NO|NUMBER|NUM|#)?\b",
        r"\bSER\s*(?:NO|NUMBER|NUM|#)?\b",
        r"\bS\s*/\s*N\b",
    ],
    "model": [
        r"\bMOTOR\s*MOD(?:EL)?\b",
        r"\bMODEL\b",
        r"\bMOD(?:EL)?\b",
    ],
    "voltage": [
        r"\bVOLTS?\b",
        r"\bVOLTAGE\b",
    ],
    "manufacture_date": [
        r"\bMFG\s*(?:DATE)?\b",
        r"\bMFD\s*(?:DATE)?\b",
        r"\bMANUFACTURE(?:D)?\s*(?:DATE)?\b",
        r"\bMANUF\s*(?:DATE)?\b",
        r"\bDATE\s*(?:OF\s*(?:MFG|MANUFACTURE))?\b",
    ],
}

STOP_PATTERNS = [
    r"\bSERIAL\s*(?:NO|NUMBER|NUM|#)?\b",
    r"\bSER\s*(?:NO|NUMBER|NUM|#)?\b",
    r"\bS\s*/\s*N\b",
    r"\bMODEL\b",
    r"\bMOTOR\s*MOD(?:EL)?\b",
    r"\bMOD(?:EL)?\b",
    r"\bVOLTS?\b",
    r"\bVOLTAGE\b",
    r"\bMFG\s*(?:DATE)?\b",
    r"\bMFD\s*(?:DATE)?\b",
    r"\bMANUFACTURE(?:D)?\s*(?:DATE)?\b",
    r"\bEQUIP(?:MENT)?\s*(?:ID|NO|NUMBER)?\b",
    r"\bASSET\s*(?:ID|NO|NUMBER)?\b",
    r"\bTAG\s*(?:ID|NO|NUMBER)?\b",
    r"\bCAT(?:ALOG)?\b",
    r"\bPART\s*(?:NO|NUMBER|#)?\b",
    r"\bP\s*/?\s*N\b",
    r"\bSIZE\b",
    r"\bLUBE\b",
    r"\bHP\b",
    r"\bAMPS?\b",
    r"\bRPM\b",
    r"\bHZ\b",
    r"\bPH\b",
    r"\bCODE\b",
    r"\bTYPE\b",
    r"\bFRAME\b",
    r"\bFR\b",
    r"\bSF\b",
    r"\bKW\b",
    r"\bINSUL\b",
    r"\bCLASS\b",
    r"\bAMB\b",
    r"\bTIME\b",
    r"\bRATING\b",
    r"\bFORM\b",
    r"\bENCL\b",
    r"\bLINE\b",
    r"\bLOW\s*VOLTAGE\b",
    r"\bHIGH\s*VOLTAGE\b",
]


@dataclass
class OCRItem:
    text: str
    conf: float
    bbox: list[list[float]]
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "task3_ocr.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


def to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in NULL_STRINGS:
        return None
    return text


def normalize_dashes(text: str) -> str:
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("‐", "-")
    )


def norm_for_label(text: str) -> str:
    """
    Normalize OCR text only for matching labels.
    Important: this is not used to alter serial numbers globally,
    because changing O/0 or I/1 blindly can corrupt serial numbers.
    """
    text = normalize_dashes(text).upper()

    replacements = {
        "M0DEL": "MODEL",
        "MOOEL": "MODEL",
        "M0D": "MOD",
        "V0LTS": "VOLTS",
        "VOLT5": "VOLTS",
        "V0LTAGE": "VOLTAGE",
        "SER1AL": "SERIAL",
        "SERI4L": "SERIAL",
        "NUM8ER": "NUMBER",
        "N0": "NO",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 50 or max_height < 30:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def auto_crop_label_region(image: np.ndarray) -> np.ndarray:
    """
    Attempts to crop or rectify the label/nameplate region.
    If no reliable rectangle is found, returns the original image.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnt_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnt_result[0] if len(cnt_result) == 2 else cnt_result[1]

    if not contours:
        return image

    image_area = float(h * w)
    best_quad = None
    best_area = 0.0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < 0.15 * image_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            quad = box.reshape(4, 2)

        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        if bw <= 0 or bh <= 0:
            continue

        aspect = max(bw / bh, bh / bw)
        if aspect < 1.2 or aspect > 12:
            continue

        if area > best_area:
            best_area = area
            best_quad = quad

    if best_quad is None:
        return image

    try:
        warped = four_point_transform(image, best_quad)

        if warped.shape[0] < 40 or warped.shape[1] < 80:
            return image

        return warped
    except Exception:
        return image


def resize_if_small(image: np.ndarray, min_width: int = 1400, max_width: int = 2600) -> np.ndarray:
    h, w = image.shape[:2]

    if w >= min_width:
        return image

    scale = min_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w > max_width:
        scale = max_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


# def deskew_image(image: np.ndarray) -> np.ndarray:
#     """
#     Uses Hough lines to estimate small rotation from horizontal label/table lines.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)

#     h, w = gray.shape[:2]
#     min_line_length = max(80, int(w * 0.25))

#     lines = cv2.HoughLinesP(
#         edges,
#         rho=1,
#         theta=np.pi / 180,
#         threshold=max(60, int(w * 0.04)),
#         minLineLength=min_line_length,
#         maxLineGap=25,
#     )

#     if lines is None:
#         return image

#     angles = []

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         dx = x2 - x1
#         dy = y2 - y1

#         if dx == 0:
#             continue

#         angle = math.degrees(math.atan2(dy, dx))

#         if -15.0 <= angle <= 15.0:
#             angles.append(angle)

#     if len(angles) < 2:
#         return image

#     median_angle = float(np.median(angles))

#     if abs(median_angle) < 0.2 or abs(median_angle) > 12:
#         return image

#     return rotate_bound(image, median_angle)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Uses Hough lines to estimate small rotation from horizontal label/table lines.
    This version is robust to different OpenCV HoughLinesP output shapes:
    - (N, 1, 4)
    - (N, 4)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape[:2]
    min_line_length = max(80, int(w * 0.25))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(60, int(w * 0.04)),
        minLineLength=min_line_length,
        maxLineGap=25,
    )

    if lines is None:
        return image

    angles = []

    for line in lines:
        # Handles both OpenCV formats:
        # line = [[x1, y1, x2, y2]]
        # line = [x1, y1, x2, y2]
        coords = np.asarray(line).reshape(-1)

        if coords.size < 4:
            continue

        x1, y1, x2, y2 = coords[:4].astype(float)

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-6:
            continue

        angle = math.degrees(math.atan2(dy, dx))

        # Only use mostly-horizontal lines.
        if -15.0 <= angle <= 15.0:
            angles.append(angle)

    if len(angles) < 2:
        return image

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.2 or abs(median_angle) > 12:
        return image

    # Rotate opposite to the detected skew angle.
    return rotate_bound(image, median_angle)


def enhance_for_ocr(image: np.ndarray, use_threshold: bool = False) -> np.ndarray:
    """
    Denoise + contrast enhancement. Thresholding is optional because
    aggressive thresholding can hurt embossed metallic nameplates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)

    if use_threshold:
        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# def preprocess_image(image: np.ndarray, use_threshold: bool = False) -> np.ndarray:
#     cropped = auto_crop_label_region(image)
#     resized = resize_if_small(cropped)
#     deskewed = deskew_image(resized)
#     enhanced = enhance_for_ocr(deskewed, use_threshold=use_threshold)
#     return enhanced

def preprocess_image(image: np.ndarray, use_threshold: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Wrapped safely so one bad preprocessing step does not crash the whole OCR run.
    """
    try:
        cropped = auto_crop_label_region(image)
    except Exception as exc:
        logging.warning("auto_crop_label_region failed: %s", exc)
        cropped = image

    try:
        resized = resize_if_small(cropped)
    except Exception as exc:
        logging.warning("resize_if_small failed: %s", exc)
        resized = cropped

    try:
        deskewed = deskew_image(resized)
    except Exception as exc:
        logging.warning("deskew_image failed: %s", exc)
        deskewed = resized

    try:
        enhanced = enhance_for_ocr(deskewed, use_threshold=use_threshold)
    except Exception as exc:
        logging.warning("enhance_for_ocr failed: %s", exc)
        enhanced = deskewed

    return enhanced

def bbox_to_item(text: str, conf: float, bbox: Any) -> OCRItem:
    pts = np.array(bbox, dtype=float)

    if pts.ndim == 1 and len(pts) == 4:
        x, y, w, h = pts
        points = np.array(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ],
            dtype=float,
        )
    else:
        points = pts.reshape(-1, 2)

    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))

    return OCRItem(
        text=str(text).strip(),
        conf=float(conf),
        bbox=points.tolist(),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx=(x1 + x2) / 2.0,
        cy=(y1 + y2) / 2.0,
    )


class OCRRunner:
    def __init__(self, engine: str = "easyocr", gpu: bool = False) -> None:
        self.engine = engine.lower().strip()

        if self.engine == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR is not installed. Install it with: pip install easyocr"
                ) from exc

            self.reader = easyocr.Reader(["en"], gpu=gpu)

        elif self.engine == "tesseract":
            try:
                import pytesseract
            except ImportError as exc:
                raise ImportError(
                    "pytesseract is not installed. Install it with: pip install pytesseract"
                ) from exc

            self.pytesseract = pytesseract

        else:
            raise ValueError("Unsupported OCR engine. Use easyocr or tesseract.")

    def read(self, image_bgr: np.ndarray) -> list[OCRItem]:
        if self.engine == "easyocr":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                decoder="greedy",
            )

            items: list[OCRItem] = []
            for bbox, text, conf in results:
                text = str(text).strip()
                if text:
                    items.append(bbox_to_item(text, conf, bbox))

            return sort_items_reading_order(items)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        data = self.pytesseract.image_to_data(
            rgb,
            config="--oem 3 --psm 6",
            output_type=self.pytesseract.Output.DICT,
        )

        items = []

        for i in range(len(data["text"])):
            text = str(data["text"][i]).strip()
            if not text:
                continue

            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0

            if conf < 0:
                continue

            x = float(data["left"][i])
            y = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])

            items.append(bbox_to_item(text, conf / 100.0, [x, y, w, h]))

        return sort_items_reading_order(items)


def sort_items_reading_order(items: list[OCRItem]) -> list[OCRItem]:
    if not items:
        return []

    heights = [item.y2 - item.y1 for item in items if item.y2 > item.y1]
    median_h = statistics.median(heights) if heights else 20.0
    y_bucket = max(10.0, median_h * 0.8)

    return sorted(items, key=lambda item: (round(item.cy / y_bucket), item.x1))


def group_items_into_lines(items: list[OCRItem]) -> list[str]:
    if not items:
        return []

    heights = [item.y2 - item.y1 for item in items if item.y2 > item.y1]
    median_h = statistics.median(heights) if heights else 20.0
    y_tol = max(12.0, median_h * 0.75)

    items_sorted = sorted(items, key=lambda item: item.cy)
    groups: list[list[OCRItem]] = []

    for item in items_sorted:
        added = False

        for group in groups:
            group_cy = statistics.mean([g.cy for g in group])
            if abs(item.cy - group_cy) <= y_tol:
                group.append(item)
                added = True
                break

        if not added:
            groups.append([item])

    lines = []

    for group in groups:
        group_sorted = sorted(group, key=lambda item: item.x1)
        line = " ".join(item.text for item in group_sorted).strip()
        if line:
            lines.append(line)

    return lines


def items_to_text(items: list[OCRItem]) -> str:
    return "\n".join(group_items_into_lines(items))


def cut_at_next_label(text: str) -> str:
    if not text:
        return text

    norm = norm_for_label(text)
    earliest = len(text)

    for pattern in STOP_PATTERNS:
        match = re.search(pattern, norm, flags=re.IGNORECASE)
        if match and match.start() > 0:
            earliest = min(earliest, match.start())

    return text[:earliest]


def clean_value(value: Any, field: str) -> str | None:
    value = to_none(value)

    if value is None:
        return None

    value = normalize_dashes(value)
    value = value.strip()
    value = re.sub(r"^[\s:;,#=_\-]+", "", value)
    value = re.sub(r"[\s:;,#=_\-]+$", "", value)

    # Remove common leading label residue.
    value = re.sub(
        r"^(NO\.?|NUMBER|NUM|#)\s*[:#=_\-]*\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    label_words = [
        "SERIAL NUMBER",
        "SERIAL NO",
        "SERIAL",
        "SER",
        "S/N",
        "MODEL",
        "MOTOR MOD",
        "MOD",
        "VOLTS",
        "VOLTAGE",
        "EQUIPMENT ID",
        "EQUIP ID",
        "ASSET ID",
        "TAG ID",
        "CATALOG",
        "CAT",
        "PART NUMBER",
        "PART NO",
        "PART",
        "P/N",
        "MFG DATE",
        "MFD DATE",
        "MFG",
        "MFD",
        "MANUFACTURED DATE",
        "MANUFACTURE DATE",
        "DATE",
    ]

    for label in label_words:
        value = re.sub(
            r"^" + re.escape(label) + r"\s*[:#=_\-]*\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

    value = value.strip()

    if not value:
        return None

    if field in {"serial_number", "model", "equipment_id"}:
        value = value.upper()
        value = value.replace(" ", "")
        value = re.sub(r"[^A-Z0-9./_\-]", "", value)
        value = value.strip("._-/")

        if value.upper() in NULL_STRINGS or not value:
            return None

        return value

    if field == "voltage":
        value = value.upper()

        # For voltage only, OCR O/I/l mistakes can safely be converted
        # because voltage should be numeric.
        value = (
            value.replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("L", "1")
        )

        value = value.replace(" TO ", "-")
        value = value.replace("TO", "-")

        match = re.search(
            r"\b\d{2,4}(?:\s*(?:-|/)\s*\d{1,4}){0,3}\b",
            value,
        )

        if not match:
            return None

        voltage = match.group(0)
        voltage = re.sub(r"\s+", "", voltage)
        return voltage

    if field == "manufacture_date":
        return normalize_date_value(value)

    return value


def normalize_date_value(value: Any) -> str | None:
    value = to_none(value)

    if value is None:
        return None

    text = normalize_dashes(str(value)).strip().upper()
    text = text.replace(".", " ")
    text = re.sub(r"\s+", " ", text)

    date_patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d{1,2}[-/]\d{4}\b",
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}\b",
        r"\b\d{4}\b",
    ]

    found = None
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            found = match.group(0)
            break

    if found is None:
        return None

    found = found.replace("SEPT", "SEP")

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%m-%d-%y",
        "%m/%d/%y",
        "%m-%Y",
        "%m/%Y",
        "%b %Y",
        "%B %Y",
        "%b %y",
        "%B %y",
        "%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(found.title(), fmt)

            if fmt == "%Y":
                return f"{dt.year:04d}"

            if fmt in {"%m-%Y", "%m/%Y", "%b %Y", "%B %Y", "%b %y", "%B %y"}:
                return f"{dt.year:04d}-{dt.month:02d}"

            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # If only a clean 4-digit year is present.
    if re.fullmatch(r"\d{4}", found):
        return found

    return None


def validate_value(value: str | None, field: str) -> bool:
    if value is None:
        return False

    if field == "serial_number":
        return 3 <= len(value) <= 50 and bool(re.search(r"\d", value))

    if field == "model":
        return 2 <= len(value) <= 50 and bool(re.search(r"[A-Z0-9]", value))

    if field == "equipment_id":
        return 2 <= len(value) <= 60 and bool(re.search(r"[A-Z0-9]", value))

    if field == "voltage":
        return bool(re.fullmatch(r"\d{2,4}(?:[-/]\d{1,4}){0,3}", value))

    if field == "manufacture_date":
        return bool(
            re.fullmatch(r"\d{4}", value)
            or re.fullmatch(r"\d{4}-\d{2}", value)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        )

    return bool(value)


def is_label_like(text: str) -> bool:
    norm = norm_for_label(text)

    for patterns in FIELD_PATTERNS.values():
        for pattern in patterns:
            if re.search(pattern, norm, flags=re.IGNORECASE):
                return True

    for pattern in STOP_PATTERNS:
        if re.search(pattern, norm, flags=re.IGNORECASE):
            return True

    return False


def extract_from_text(field: str, full_text: str) -> str | None:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    for line in lines:
        norm_line = norm_for_label(line)

        # Skip wiring-diagram headings such as LOW VOLTAGE / HIGH VOLTAGE.
        if field == "voltage" and re.search(r"\b(LOW|HIGH)\s+VOLTAGE\b", norm_line):
            continue

        for pattern in FIELD_PATTERNS[field]:
            match = re.search(pattern, norm_line, flags=re.IGNORECASE)

            if not match:
                continue

            after = line[match.end() :]
            after = cut_at_next_label(after)

            candidate = clean_value(after, field)

            if validate_value(candidate, field):
                return candidate

    return None


def match_label_for_field(text: str, field: str) -> bool:
    norm = norm_for_label(text)

    if field == "voltage" and re.search(r"\b(LOW|HIGH)\s+VOLTAGE\b", norm):
        return False

    for pattern in FIELD_PATTERNS[field]:
        if re.search(pattern, norm, flags=re.IGNORECASE):
            return True

    return False


def extract_spatially(
    field: str,
    items: list[OCRItem],
    image_shape: tuple[int, int, int],
    min_conf: float,
) -> str | None:
    """
    Uses OCR bounding boxes. If a label like SERIAL appears on the left,
    the nearest text on the same row to the right is treated as the value.
    """
    if not items:
        return None

    img_h, img_w = image_shape[:2]

    label_items = [
        item
        for item in items
        if item.conf >= min_conf and match_label_for_field(item.text, field)
    ]

    for label in label_items:
        # Case: same OCR item contains label + value.
        same_item_candidate = extract_from_text(field, label.text)
        if validate_value(same_item_candidate, field):
            return same_item_candidate

        label_h = max(8.0, label.y2 - label.y1)
        y_tol = max(16.0, label_h * 0.95, img_h * 0.025)

        right_items = []

        for item in items:
            if item is label:
                continue
            if item.conf < min_conf:
                continue

            same_row = abs(item.cy - label.cy) <= y_tol
            to_right = item.cx > label.cx and item.x1 >= label.x1 - 5

            if same_row and to_right:
                right_items.append(item)

        right_items = sorted(right_items, key=lambda item: item.x1)

        pieces = []

        for item in right_items[:6]:
            item_is_label = is_label_like(item.text)

            # For "SERIAL NUMBER", NUMBER may appear between label and value.
            allowed_serial_word = (
                field == "serial_number"
                and norm_for_label(item.text).strip() in {"NUMBER", "NO", "NUM"}
            )

            if pieces and item_is_label and not allowed_serial_word:
                break

            pieces.append(item.text)

            joined = " ".join(pieces)
            joined = cut_at_next_label(joined)
            candidate = clean_value(joined, field)

            if validate_value(candidate, field):
                return candidate

        # Also check a value directly below the label, useful for table layouts.
        below_items = []

        for item in items:
            if item is label:
                continue
            if item.conf < min_conf:
                continue

            below = label.cy < item.cy < label.cy + max(80.0, img_h * 0.12)
            horizontal_overlap = not (item.x2 < label.x1 or item.x1 > label.x2)

            if below and horizontal_overlap:
                below_items.append(item)

        below_items = sorted(below_items, key=lambda item: (item.cy, item.x1))

        pieces = []

        for item in below_items[:4]:
            if pieces and is_label_like(item.text):
                break

            pieces.append(item.text)
            joined = " ".join(pieces)
            joined = cut_at_next_label(joined)
            candidate = clean_value(joined, field)

            if validate_value(candidate, field):
                return candidate

    return None


def extract_fields(
    items: list[OCRItem],
    image_shape: tuple[int, int, int],
    min_conf: float,
) -> dict[str, str | None]:
    full_text = items_to_text(items)

    extracted: dict[str, str | None] = {}

    for field in FIELDS:
        value = extract_from_text(field, full_text)

        if not validate_value(value, field):
            value = extract_spatially(field, items, image_shape, min_conf=min_conf)

        if not validate_value(value, field):
            value = None

        extracted[field] = value

    return extracted


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)

            current.append(min(insert_cost, delete_cost, replace_cost))

        previous = current

    return previous[-1]


def canonical_for_compare(value: Any, field: str) -> str | None:
    value = clean_value(value, field)

    if value is None:
        return None

    if field in {"serial_number", "model", "equipment_id"}:
        value = value.upper()
        value = normalize_dashes(value)
        value = re.sub(r"\s+", "", value)
        return value

    return value


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_by_image = {}
    pred_by_basename = {}

    for pred in predictions:
        image_key = str(pred["image"])
        pred_by_image[image_key] = pred
        pred_by_basename[Path(image_key).name] = pred

    total = 0
    correct = 0
    non_null_total = 0
    non_null_correct = 0

    by_field = {
        field: {
            "total": 0,
            "correct": 0,
            "non_null_gt_total": 0,
            "non_null_gt_correct": 0,
        }
        for field in FIELDS
    }

    per_image = []

    serial_edit_sum = 0
    serial_char_sum = 0
    serial_cers = []

    for gt in gt_rows:
        image_key = str(gt["image"])
        pred = pred_by_image.get(image_key) or pred_by_basename.get(Path(image_key).name)

        if pred is None:
            pred = {"image": image_key, **{field: None for field in FIELDS}}

        image_result = {
            "image": image_key,
            "fields": {},
            "correct": 0,
            "total": len(FIELDS),
        }

        for field in FIELDS:
            gt_value = canonical_for_compare(gt.get(field), field)
            pred_value = canonical_for_compare(pred.get(field), field)

            is_correct = gt_value == pred_value

            total += 1
            by_field[field]["total"] += 1

            if is_correct:
                correct += 1
                by_field[field]["correct"] += 1
                image_result["correct"] += 1

            if gt_value is not None:
                non_null_total += 1
                by_field[field]["non_null_gt_total"] += 1

                if is_correct:
                    non_null_correct += 1
                    by_field[field]["non_null_gt_correct"] += 1

            image_result["fields"][field] = {
                "gt": gt_value,
                "pred": pred_value,
                "correct": is_correct,
            }

        gt_serial = canonical_for_compare(gt.get("serial_number"), "serial_number")
        pred_serial = canonical_for_compare(pred.get("serial_number"), "serial_number")

        if gt_serial:
            pred_serial_for_cer = pred_serial or ""
            edit_distance = levenshtein_distance(pred_serial_for_cer, gt_serial)
            cer = safe_div(edit_distance, len(gt_serial))

            serial_edit_sum += edit_distance
            serial_char_sum += len(gt_serial)
            serial_cers.append(cer)

            image_result["serial_number_cer"] = cer

        per_image.append(image_result)

    metrics = {
        "overall_field_accuracy_including_null": safe_div(correct, total),
        "overall_field_accuracy_non_null_gt": safe_div(non_null_correct, non_null_total),
        "total_field_comparisons": total,
        "correct_field_comparisons": correct,
        "non_null_gt_field_comparisons": non_null_total,
        "correct_non_null_gt_field_comparisons": non_null_correct,
        "by_field": {},
        "serial_number_cer": {
            "aggregate": safe_div(serial_edit_sum, serial_char_sum),
            "mean_per_image": float(np.mean(serial_cers)) if serial_cers else 0.0,
            "num_serials_evaluated": len(serial_cers),
        },
        "per_image": per_image,
    }

    for field, counts in by_field.items():
        metrics["by_field"][field] = {
            **counts,
            "accuracy_including_null": safe_div(counts["correct"], counts["total"]),
            "accuracy_non_null_gt": safe_div(
                counts["non_null_gt_correct"],
                counts["non_null_gt_total"],
            ),
        }

    return metrics


def load_ground_truth(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Ground truth CSV is empty: {csv_path}")

    possible_image_columns = [
        "image",
        "image_name",
        "filename",
        "file",
        "path",
        "image_path",
    ]

    image_col = None

    for col in possible_image_columns:
        if col in rows[0]:
            image_col = col
            break

    if image_col is None:
        raise ValueError(
            "CSV must contain one image filename column. "
            "Accepted names: image, image_name, filename, file, path, image_path"
        )

    normalized_rows = []

    for row in rows:
        item = {"image": row.get(image_col, "").strip()}

        for field in FIELDS:
            item[field] = row.get(field, "")

        normalized_rows.append(item)

    return normalized_rows


def find_image_path(image_value: str, images_dir: Path, gt_csv: Path) -> Path:
    image_value = str(image_value).strip()
    p = Path(image_value)

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                images_dir / p,
                gt_csv.parent / p,
                Path.cwd() / p,
                Path.cwd() / "task3" / "data" / "images" / p,
                Path.cwd() / "data" / "task3" / "images" / p,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last attempt: recursive basename search inside images_dir.
    if images_dir.exists():
        matches = list(images_dir.rglob(Path(image_value).name))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not locate image: {image_value}")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"OpenCV failed to read image: {path}")

    return image


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown_report(
    path: Path,
    raw_metrics: dict[str, Any],
    pre_metrics: dict[str, Any],
    runtime_info: dict[str, Any],
) -> None:
    lines = []

    lines.append("# Task 3 OCR Accuracy Report")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- OCR engine: `{runtime_info['ocr_engine']}`")
    lines.append(f"- Python: `{runtime_info['python']}`")
    lines.append(f"- Platform: `{runtime_info['platform']}`")
    lines.append(f"- OpenCV: `{runtime_info['opencv']}`")
    lines.append(f"- Total images: `{runtime_info['num_images']}`")
    lines.append(f"- Runtime seconds: `{runtime_info['runtime_seconds']:.2f}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Mode | Field Accuracy incl. nulls | Field Accuracy non-null GT | Serial CER aggregate | Serial CER mean/image |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Raw OCR | {percent(raw_metrics['overall_field_accuracy_including_null'])} "
        f"| {percent(raw_metrics['overall_field_accuracy_non_null_gt'])} "
        f"| {raw_metrics['serial_number_cer']['aggregate']:.4f} "
        f"| {raw_metrics['serial_number_cer']['mean_per_image']:.4f} |"
    )
    lines.append(
        f"| Preprocessed OCR | {percent(pre_metrics['overall_field_accuracy_including_null'])} "
        f"| {percent(pre_metrics['overall_field_accuracy_non_null_gt'])} "
        f"| {pre_metrics['serial_number_cer']['aggregate']:.4f} "
        f"| {pre_metrics['serial_number_cer']['mean_per_image']:.4f} |"
    )
    lines.append("")

    lines.append("## Field-Level Accuracy by Field")
    lines.append("")
    lines.append(
        "| Field | Raw incl. nulls | Preprocessed incl. nulls | Raw non-null GT | Preprocessed non-null GT |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    for field in FIELDS:
        raw_f = raw_metrics["by_field"][field]
        pre_f = pre_metrics["by_field"][field]

        lines.append(
            f"| `{field}` "
            f"| {percent(raw_f['accuracy_including_null'])} "
            f"| {percent(pre_f['accuracy_including_null'])} "
            f"| {percent(raw_f['accuracy_non_null_gt'])} "
            f"| {percent(pre_f['accuracy_non_null_gt'])} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `labels_output.json` contains the final structured output from the preprocessed OCR path."
    )
    lines.append(
        "- `labels_output_raw.json` contains extraction results without OpenCV preprocessing."
    )
    lines.append(
        "- Empty ground-truth values are treated as `null`. Accuracy is reported both including nulls and only for non-null ground-truth fields."
    )
    lines.append(
        "- Serial-number CER is computed only on rows where ground truth serial number is available."
    )
    lines.append("")

    write_text(path, "\n".join(lines))


def make_template_csv(images_dir: Path, output_csv: Path) -> None:
    images = sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not images:
        raise ValueError(f"No images found in {images_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *FIELDS])
        writer.writeheader()

        for image in images:
            writer.writerow(
                {
                    "image": image.name,
                    "equipment_id": "",
                    "serial_number": "",
                    "model": "",
                    "voltage": "",
                    "manufacture_date": "",
                }
            )

    logging.info("Template CSV written to %s", output_csv)


def process_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)

    start_time = time.time()

    images_dir = Path(args.images_dir)
    gt_csv = Path(args.ground_truth)

    logging.info("Task 3 OCR started")
    logging.info("Images directory: %s", images_dir)
    logging.info("Ground truth CSV: %s", gt_csv)
    logging.info("Output directory: %s", out_dir)
    logging.info("OCR engine: %s", args.engine)
    logging.info("Use threshold preprocessing: %s", args.threshold)

    if args.make_template:
        make_template_csv(images_dir, gt_csv)
        return

    gt_rows = load_ground_truth(gt_csv)

    preprocessed_dir = out_dir / "preprocessed"
    raw_text_dir = out_dir / "ocr_text" / "raw"
    pre_text_dir = out_dir / "ocr_text" / "preprocessed"

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    pre_text_dir.mkdir(parents=True, exist_ok=True)

    ocr = OCRRunner(engine=args.engine, gpu=args.gpu)

    raw_predictions = []
    pre_predictions = []

    for idx, row in enumerate(gt_rows, start=1):
        image_key = row["image"]

        logging.info("[%d/%d] Processing %s", idx, len(gt_rows), image_key)

        image_path = find_image_path(image_key, images_dir, gt_csv)
        image = read_image(image_path)

        raw_items = ocr.read(image)
        raw_text = items_to_text(raw_items)
        raw_fields = extract_fields(raw_items, image.shape, min_conf=args.min_conf)

        raw_record = {"image": image_key, **raw_fields}
        raw_predictions.append(raw_record)

        preprocessed = preprocess_image(image, use_threshold=args.threshold)

        pre_name = Path(image_key).stem + "_preprocessed.png"
        pre_path = preprocessed_dir / pre_name
        cv2.imwrite(str(pre_path), preprocessed)

        pre_items = ocr.read(preprocessed)
        pre_text = items_to_text(pre_items)
        pre_fields = extract_fields(pre_items, preprocessed.shape, min_conf=args.min_conf)

        pre_record = {"image": image_key, **pre_fields}
        pre_predictions.append(pre_record)

        write_text(raw_text_dir / f"{Path(image_key).stem}.txt", raw_text)
        write_text(pre_text_dir / f"{Path(image_key).stem}.txt", pre_text)

    save_json(out_dir / "labels_output_raw.json", raw_predictions)
    save_json(out_dir / "labels_output.json", pre_predictions)

    raw_metrics = evaluate_predictions(raw_predictions, gt_rows)
    pre_metrics = evaluate_predictions(pre_predictions, gt_rows)

    runtime_seconds = time.time() - start_time

    runtime_info = {
        "ocr_engine": args.engine,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "num_images": len(gt_rows),
        "runtime_seconds": runtime_seconds,
        "preprocessing": {
            "auto_crop_or_rectify": True,
            "resize_if_small": True,
            "deskew": True,
            "denoise": True,
            "clahe_contrast": True,
            "adaptive_threshold": bool(args.threshold),
        },
    }

    report = {
        "runtime": runtime_info,
        "raw": raw_metrics,
        "preprocessed": pre_metrics,
    }

    save_json(out_dir / "accuracy_report.json", report)

    write_markdown_report(
        out_dir / "accuracy_report.md",
        raw_metrics=raw_metrics,
        pre_metrics=pre_metrics,
        runtime_info=runtime_info,
    )

    logging.info("Done")
    logging.info("Final structured output: %s", out_dir / "labels_output.json")
    logging.info("Accuracy report JSON: %s", out_dir / "accuracy_report.json")
    logging.info("Accuracy report MD: %s", out_dir / "accuracy_report.md")
    logging.info(
        "Preprocessed field accuracy non-null GT: %.4f",
        pre_metrics["overall_field_accuracy_non_null_gt"],
    )
    logging.info(
        "Preprocessed serial CER aggregate: %.4f",
        pre_metrics["serial_number_cer"]["aggregate"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 3 OCR pipeline with preprocessing, structured extraction and accuracy reporting."
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default="task3/data/images",
        help="Directory containing label/nameplate images.",
    )

    parser.add_argument(
        "--ground_truth",
        type=str,
        default="task3/data/labels_ground_truth.csv",
        help="Path to labels_ground_truth.csv.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="task3/outputs",
        help="Output directory.",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract"],
        help="OCR engine to use.",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR if available.",
    )

    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Apply adaptive thresholding in preprocessing. Useful for clean printed labels, but may hurt embossed plates.",
    )

    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.10,
        help="Minimum OCR confidence for spatial extraction.",
    )

    parser.add_argument(
        "--make_template",
        action="store_true",
        help="Create an empty ground-truth CSV template from images_dir and exit.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
