"""Utility functions for reading and converting images."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image_from_stream(stream) -> np.ndarray | None:
    """Read an uploaded file stream into a BGR OpenCV image.

    Args:
        stream: A file-like object (e.g. ``request.FILES["field"]``).

    Returns:
        A NumPy BGR image array, or ``None`` on failure.
    """
    try:
        data = stream.read()
        if not data:
            logger.warning("Empty image stream received")
            return None
        image = np.asarray(bytearray(data), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("cv2.imdecode returned None — invalid image data")
        return image
    except Exception:
        logger.exception("Failed to read image from stream")
        return None


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale, handling already-gray inputs."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
