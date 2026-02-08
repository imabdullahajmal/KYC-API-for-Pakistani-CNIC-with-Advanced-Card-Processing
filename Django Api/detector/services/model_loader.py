"""Singleton loaders for the YOLOv8 detector and EasyOCR reader.

Models are heavy — loading them once and reusing across requests is critical
for acceptable response times.
"""

import logging
import os

import easyocr

from yolov8 import YOLOv8

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve the ONNX model path relative to the Django project root.
# Default: ``<repo>/Model/best.onnx`` (one level above ``Django Api/``).
# Override with the ``ONNX_MODEL_PATH`` environment variable.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "Model", "best.onnx",
)
MODEL_PATH: str = os.environ.get("ONNX_MODEL_PATH", _DEFAULT_MODEL_PATH)

# Module-level singletons — initialised lazily on first access.
_yolo_model: YOLOv8 | None = None
_ocr_reader: easyocr.Reader | None = None


def get_yolo_model(conf_thres: float = 0.2, iou_thres: float = 0.3) -> YOLOv8:
    """Return a shared YOLOv8 instance (created once)."""
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLOv8 model from %s", MODEL_PATH)
        _yolo_model = YOLOv8(MODEL_PATH, conf_thres=conf_thres, iou_thres=iou_thres)
    return _yolo_model


def get_ocr_reader(languages: list[str] | None = None, gpu: bool = False) -> easyocr.Reader:
    """Return a shared EasyOCR Reader instance (created once)."""
    global _ocr_reader
    if _ocr_reader is None:
        langs = languages or ["en"]
        logger.info("Initialising EasyOCR reader (languages=%s, gpu=%s)", langs, gpu)
        _ocr_reader = easyocr.Reader(langs, gpu=gpu)
    return _ocr_reader
