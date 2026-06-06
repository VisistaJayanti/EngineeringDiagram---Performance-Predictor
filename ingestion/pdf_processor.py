"""
ingestion/pdf_processor.py
--------------------------
Converts an uploaded file (PDF or image) into a list of
normalized page images ready for the VLM pipeline.

Step by step:
    1. Check if input is PDF or image
    2. If PDF  → rasterize every page at 300 DPI using PyMuPDF
    3. If image → load directly, read DPI from EXIF if available
    4. Return a list of dicts, one per page

Why 300 DPI:
    Engineering drawings have dimension text as small as 2mm tall.
    At 96 DPI (screen resolution) that becomes 8 pixels — unreadable.
    At 300 DPI it becomes 24 pixels — clear enough for OCR and VLM.

Dependencies:
    pip install pymupdf pillow
"""

import io
import base64
import tempfile
import os
from pathlib import Path

import fitz                          # PyMuPDF
from PIL import Image

from config.settings import TARGET_DPI
from utils.image_utils import pil_to_b64


def process_upload(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Main entry point. Takes raw file bytes from an upload and returns
    a list of page dicts.

    Args:
        file_bytes : raw bytes of the uploaded file
        filename   : original filename, used to detect PDF vs image

    Returns:
        list of dicts, one per page:
        {
            "page_number" : int,
            "image_b64"   : str,   base64 PNG
            "width_px"    : int,
            "height_px"   : int,
            "dpi"         : int,
        }
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_to_images(file_bytes)
    elif suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
        return _image_to_page(file_bytes, suffix)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Accepted: .pdf .jpg .jpeg .png .tif .tiff .bmp"
        )


def _pdf_to_images(file_bytes: bytes) -> list[dict]:
    """
    Rasterize every page of a PDF at TARGET_DPI.

    Why PyMuPDF (fitz) over pdf2image:
        - No Ghostscript dependency
        - Faster rasterization of vector CAD line work
        - Direct DPI control via zoom matrix
        - Handles embedded fonts in title blocks correctly
    """
    # PyMuPDF needs a file path, not bytes — write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    pages = []
    try:
        doc = fitz.open(tmp_path)

        # PDF default resolution is 72 DPI
        # zoom = TARGET_DPI / 72 scales up to our desired resolution
        zoom   = TARGET_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render to RGB pixmap — no alpha channel needed for CAD
            pixmap = page.get_pixmap(matrix=matrix, alpha=False,
                                     colorspace=fitz.csRGB)

            # Convert pixmap bytes → PIL Image
            img = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples
            )

            pages.append({
                "page_number" : page_num + 1,
                "image_b64"   : pil_to_b64(img),
                "width_px"    : pixmap.width,
                "height_px"   : pixmap.height,
                "dpi"         : TARGET_DPI,
            })

        doc.close()

    finally:
        # Always clean up the temp file even if something crashed
        os.unlink(tmp_path)

    return pages


def _image_to_page(file_bytes: bytes, suffix: str) -> list[dict]:
    """
    Load a JPG / PNG / TIFF image file as a single page.

    We cannot know the true DPI of a raster image unless the EXIF
    data says so. We read it if present, otherwise assume 96 DPI
    (standard screen resolution) and flag it.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Read DPI from EXIF — present in scanned TIFFs, sometimes JPEGs
    exif_dpi = img.info.get("dpi", (96, 96))
    if isinstance(exif_dpi, tuple):
        dpi = int(exif_dpi[0])
    else:
        dpi = int(exif_dpi)

    return [{
        "page_number" : 1,
        "image_b64"   : pil_to_b64(img),
        "width_px"    : img.width,
        "height_px"   : img.height,
        "dpi"         : dpi,
    }]