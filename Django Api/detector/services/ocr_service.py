"""OCR and QR-code extraction logic for Pakistani CNIC cards.

All heavy image-processing helpers live here so that ``views.py`` stays thin.
"""

import csv
import logging
from datetime import datetime

import cv2
import numpy as np
from pyzbar.pyzbar import decode

from .image_utils import to_grayscale
from .model_loader import get_ocr_reader, get_yolo_model

logger = logging.getLogger(__name__)

# Phrases commonly printed on CNIC cards that should be filtered from OCR.
_COMMON_FRONT = frozenset([
    "/", "Pakistan", "pakistan", "Name", "Father Name", "Gender",
    "Country of Stay", "Identity Number", "Date of Issue", "Date of Expiry",
    "Signature", "PAKISTAN", "ISLAMIC REpUBLIC OF PAKISTAN",
])

_COMMON_LOWER = frozenset([
    "Date of Expiry", "United Arab Emirates", "Date", "of Expiry",
    "Date of Birth", "/", "Pakistan",
])


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_card_regions(front_gray: np.ndarray, back_image: np.ndarray):
    """Run YOLOv8 on both card sides and return annotated ROIs.

    Returns:
        ``(back_roi, front_roi)`` — images with bounding-box overlays.
    """
    model = get_yolo_model()

    boxes, scores, class_ids = model(front_gray)
    front_roi = model.draw_detections(front_gray)

    boxes, scores, class_ids = model(back_image)
    back_roi = model.draw_detections(back_image)

    return back_roi, front_roi


def ocr_front_upper(front_image: np.ndarray, faces) -> tuple[dict, list[str]]:
    """Extract name / guardian name from the upper-left region of the front card.

    Returns:
        ``(structured_dict, raw_texts)``
    """
    model = get_yolo_model()
    reader = get_ocr_reader()

    try:
        boxes, _, _ = model(front_image)
        if boxes is None or len(boxes) == 0:
            return {}, []

        if len(faces) == 0:
            return {}, []

        # Crop the region left of the face
        upper_roi = None
        for (x, y, w, h) in faces:
            left_x = max(0, x - 410)
            left_y = max(0, y - 72)
            right_x = max(0, x - 15)
            bottom_y = y + h + 40
            upper_roi = front_image[left_y:bottom_y, left_x:right_x]

        if upper_roi is None or upper_roi.size == 0:
            return {}, []

        gray_roi = to_grayscale(upper_roi)
        result = reader.readtext(gray_roi)

        filtered = [
            (det[0], det[1].replace("-", ""))
            for det in result
            if det[1] not in _COMMON_FRONT
        ]
        texts = [t[1] for t in filtered]

        data: dict = {}
        if len(texts) >= 2:
            data = {"Name": texts[0], "Guardian Name": texts[1]}
        elif len(texts) == 1:
            data = {"Name": texts[0]}

        return data, texts

    except Exception:
        logger.exception("ocr_front_upper failed")
        return {}, []


def ocr_front_lower(front_gray: np.ndarray, faces) -> tuple[str | None, dict, list[str]]:
    """Extract CNIC number, DoB, issue/expiry dates from the lower-left region.

    Returns:
        ``(cnic_number, structured_dict, raw_texts)``
    """
    reader = get_ocr_reader()

    try:
        if len(faces) == 0:
            return None, {}, []

        x, y, w, h = faces[0]
        left_x = max(0, x - 410)
        top_y = y + 110
        right_x = max(0, x - 15)
        bottom_y = y + h + 135

        roi = front_gray[top_y:bottom_y, left_x:right_x]
        if roi is None or roi.size == 0:
            return None, {}, []

        result = reader.readtext(roi)

        filtered = [
            (det[0], det[1].replace("-", ""))
            for det in result
            if det[1] not in _COMMON_LOWER
        ]
        texts = [t[1] for t in filtered]
        texts = _convert_dates(texts)

        data: dict = {}
        if len(texts) >= 4:
            data = {
                "Id Card Number": texts[0],
                "Date Of Birth": texts[1],
                "Date Of Issue": texts[2],
                "Date Of Expiry": texts[3],
            }

        # Find the CNIC-shaped string (xxxxx-xxxxxxx-x → 15 chars with dashes)
        cnic = None
        for det in result:
            raw = det[1]
            if "-" in raw and len(raw) == 15:
                cnic = raw.replace("-", "")
                break

        return cnic, data, texts

    except Exception:
        logger.exception("ocr_front_lower failed")
        return None, {}, []


def ocr_back_qr(back_roi: np.ndarray) -> str | None:
    """Decode the QR code on the back of the CNIC and return the CNIC number."""
    model = get_yolo_model()

    try:
        model(back_roi)  # run detection (unused boxes, but keeps pipeline consistent)
        gray = to_grayscale(back_roi)
        qr_codes = decode(gray)
        if qr_codes:
            raw = qr_codes[0].data.decode("utf-8")
            return raw[12:25]
        return None
    except Exception:
        logger.exception("ocr_back_qr failed")
        return None


def validate_cnic(front_cnic: str | None, back_cnic: str | None) -> bool:
    """Return ``True`` when both sides yield the same CNIC string."""
    if front_cnic is None or back_cnic is None:
        return False
    return front_cnic == back_cnic


def merge_card_data(
    upper: dict,
    lower: dict,
    raw_upper: list[str],
    raw_lower: list[str],
    csv_path: str | None = "IdCardData.csv",
) -> dict:
    """Merge upper/lower dicts, sanitise values, and optionally persist to CSV.

    Returns:
        The merged, sanitised dictionary.
    """
    merged: dict = {**(upper or {}), **(lower or {})}
    for key in list(merged):
        if isinstance(merged[key], str):
            merged[key] = merged[key].replace(".", "")

    if csv_path:
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["key", "value"])
                for k, v in merged.items():
                    writer.writerow([k, v])
        except Exception:
            logger.exception("CSV write failed (non-fatal)")

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convert_dates(elements: list[str]) -> list[str]:
    """Convert ``DD.MM.YYYY`` strings to ``DDMMYY``; leave others unchanged."""
    out = []
    for el in elements:
        if not isinstance(el, str):
            out.append(el)
            continue
        try:
            dt = datetime.strptime(el, "%d.%m.%Y")
            out.append(dt.strftime("%d%m%y"))
        except ValueError:
            out.append(el)
    return out
