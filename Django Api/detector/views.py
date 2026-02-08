"""API views for CNIC card detection and OCR.

This module is intentionally thin — all heavy lifting lives in
``detector.services``.
"""

import logging

import cv2
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .services.image_utils import read_image_from_stream, to_grayscale
from .services.ocr_service import (
    detect_card_regions,
    merge_card_data,
    ocr_back_qr,
    ocr_front_lower,
    ocr_front_upper,
    validate_cnic,
)

logger = logging.getLogger(__name__)


def _envelope(*, success: bool = False, message: str = "", data=None, errors: list | None = None):
    """Return a consistent JSON response envelope."""
    return {
        "success": success,
        "message": message,
        "data": data,
        "errors": errors or [],
    }


@api_view(["POST"])
def id_detector(request):
    """Accept front + back CNIC images, run detection/OCR, and return structured JSON."""

    try:
        # ---- validate uploads ------------------------------------------------
        if "front_image" not in request.FILES or "back_image" not in request.FILES:
            return Response(
                _envelope(message="Missing required files: 'front_image' and 'back_image'",
                          errors=["missing_files"]),
                status=status.HTTP_400_BAD_REQUEST,
            )

        front_image = read_image_from_stream(request.FILES["front_image"])
        back_image = read_image_from_stream(request.FILES["back_image"])

        if front_image is None or back_image is None:
            return Response(
                _envelope(message="Unable to decode one or both uploaded images",
                          errors=["invalid_image"]),
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ---- face detection ---------------------------------------------------
        front_gray = to_grayscale(front_image)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            front_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) == 0:
            return Response(
                _envelope(message="No face detected on the front image",
                          errors=["no_face_detected"]),
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # ---- card region detection --------------------------------------------
        back_roi, front_roi = detect_card_regions(front_gray, back_image)

        # ---- OCR --------------------------------------------------------------
        upper_data, upper_raw = ocr_front_upper(front_image, faces)
        cnic_front, lower_data, lower_raw = ocr_front_lower(front_gray, faces)
        cnic_back = ocr_back_qr(back_roi)

        # ---- CNIC cross-validation -------------------------------------------
        if not validate_cnic(cnic_front, cnic_back):
            return Response(
                _envelope(
                    message="Front and back CNIC numbers do not match",
                    errors=["cnic_mismatch"],
                    data={"front_cnic": cnic_front, "back_cnic": cnic_back},
                ),
                status=status.HTTP_406_NOT_ACCEPTABLE,
            )

        # ---- merge & respond -------------------------------------------------
        merged = merge_card_data(upper_data, lower_data, upper_raw, lower_raw)

        return Response(
            _envelope(
                success=True,
                message="ID card processed successfully",
                data={
                    "card_info": merged,
                    "front_cnic": cnic_front,
                    "back_cnic": cnic_back,
                    "ocr_upper_raw": upper_raw,
                    "ocr_lower_raw": lower_raw,
                },
            ),
            status=status.HTTP_200_OK,
        )

    except Exception as exc:
        logger.exception("Unhandled error in id_detector")
        return Response(
            _envelope(message="Internal server error", errors=[str(exc)]),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
