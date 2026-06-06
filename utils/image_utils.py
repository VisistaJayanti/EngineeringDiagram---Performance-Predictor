import io
import base64
from PIL import Image


def pil_to_b64(img: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def b64_to_pil(image_b64: str) -> Image.Image:
    """Convert a base64 string back to a PIL image."""
    data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(data)).convert("RGB")