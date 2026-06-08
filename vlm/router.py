"""
vlm/router.py
-------------
Loads both models once and routes tasks to the right one.

    InternVL2   → geometry / feature extraction
    Qwen2.5-VL  → annotation / text extraction

Both models are cached after first load using lru_cache.
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

#AFTER RUNNING GEOMETRY EXTRACTION, NOW DOING POST-PROCESSING WHAT IS DOES IS FILTER HALLUCINATED FEATURES SO IT IS EASIER FOR GETTING ONLY ACCURATE FEATURES 
def _filter_hallucinated_features(features: list[dict]) -> list[dict]:
    """
    Detect and remove hallucinated template responses from InternVL2.

    InternVL2-8B tends to generate grid patterns instead of reading
    the actual image. Two patterns detected:

    Pattern A — vertical grid:
        cy values are evenly spaced: 0.1, 0.2, 0.3...
        All steps between sorted cy values are equal.

    Pattern B — diagonal grid:
        Both cx and cy increment together: (0.1,0.1), (0.2,0.2)...
        cx == cy for most features.

    Pattern C — boundary overflow:
        Features at cx or cy >= 0.98 are outside image.

    When a grid pattern is detected, return empty list and let the
    VLM pass in spatial_aligner handle linking from annotations alone.
    """
    if not features:
        return features

    # Step 1 — normalize coordinates
    for f in features:
        loc = f.get("location", {})
        cx  = float(loc.get("cx", 0.5))
        cy  = float(loc.get("cy", 0.5))
        if cx > 1.0: cx = cx / 1024.0
        if cy > 1.0: cy = cy / 1024.0
        f["location"] = {"cx": round(cx, 3), "cy": round(cy, 3)}

    # Step 2 — boundary filter
    valid = [
        f for f in features
        if f["location"]["cx"] < 0.98
        and f["location"]["cy"] < 0.98
    ]

    if len(valid) < 3:
        return valid

    # Step 3 — detect vertical grid pattern (cy steps equal)
    cy_values = sorted([f["location"]["cy"] for f in valid])
    if len(cy_values) >= 4:
        steps = [round(cy_values[i+1] - cy_values[i], 2)
                 for i in range(len(cy_values) - 1)]
        unique_steps = set(steps)
        if len(unique_steps) == 1:
            step = list(unique_steps)[0]
            if 0.08 <= step <= 0.12:
                print(f"[Router] Vertical grid detected (step={step}) "
                      f"— model hallucinating, returning empty")
                return []

    # Step 4 — detect diagonal grid pattern (cx == cy)
    diagonal_count = sum(
        1 for f in valid
        if abs(f["location"]["cx"] - f["location"]["cy"]) < 0.05
    )
    if diagonal_count >= len(valid) * 0.7:
        print(f"[Router] Diagonal grid detected "
              f"({diagonal_count}/{len(valid)} features on diagonal) "
              f"— model hallucinating, returning empty")
        return []

    # Step 5 — detect all-same description (another hallucination sign)
    descriptions = [f.get("description", "") for f in valid]
    if len(set(descriptions)) == 1 and len(valid) > 2:
        print(f"[Router] All features have identical description "
              f"— model hallucinating, returning empty")
        return []

    removed = len(features) - len(valid)
    if removed > 0:
        print(f"[Router] Filtered {removed} boundary features")
    print(f"[Router] Features after filter: {len(valid)}/{len(features)}")
    return valid

def run_annotation_extraction(image_b64: str) -> list[dict]:
    """Extract text annotations from one tile using Qwen2.5-VL."""
    model = get_text_model()
    return model.extract_annotations(image_b64)