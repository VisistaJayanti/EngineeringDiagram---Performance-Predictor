#IMPORTING THE PACKAGES 
import io
import base64
import tempfile
import os
from pathlib import Path

import fitz                          
from PIL import Image

from config.settings import TARGET_DPI
from utils.image_utils import pil_to_b64


def process_upload(file_bytes: bytes, filename: str) -> list[dict]:
   
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