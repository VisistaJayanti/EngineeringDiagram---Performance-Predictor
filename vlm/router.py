"""
vlm/router.py
-------------
Loads both models once and routes tasks to the right one.

    InternVL2   → geometry / feature extraction
    Qwen2.5-VL  → annotation / text extraction

Both models are cached after first load.
Call get_geometry_model() and get_text_model() anywhere in the pipeline.
"""

from functools import lru_cache
from vlm.internvl2 import InternVL2
from vlm.qwen25vl  import Qwen25VL


@lru_cache(maxsize=1)
def get_geometry_model() -> InternVL2:
    """
    Return InternVL2 instance.
    Loaded once on first call, cached for all subsequent calls.
    """
    return InternVL2()


@lru_cache(maxsize=1)
def get_text_model() -> Qwen25VL:
    """
    Return Qwen2.5-VL instance.
    Loaded once on first call, cached for all subsequent calls.
    """
    return Qwen25VL()


def run_geometry_extraction(image_b64: str) -> list[dict]:
    """Extract geometric features from one tile using InternVL2."""
    model = get_geometry_model()
    return model.extract_features(image_b64)


def run_annotation_extraction(image_b64: str) -> list[dict]:
    """Extract text annotations from one tile using Qwen2.5-VL."""
    model = get_text_model()
    return model.extract_annotations(image_b64)