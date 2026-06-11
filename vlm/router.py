"""
vlm/router.py
-------------
Loads both models once and routes tasks to the right one.

    InternVL2   → geometry / feature extraction
    Qwen2.5-VL  → annotation / text extraction

Both models are cached after first load using lru_cache.
"""

from functools import lru_cache
from vlm.kimik2 import KimiVLM
from vlm.qwen25vl  import Qwen25VL


@lru_cache(maxsize=1)
def get_geometry_model() -> KimiVLM:
    """
    Return InternVL2 instance.
    Loaded once on first call, cached for all subsequent calls.
    """
    return KimiVLM()


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
    #Defining the features
    features = model.extract_features(image_b64)
    return _filter_hallucinated_features(features)

#AFTER RUNNING GEOMETRY EXTRACTION, NOW DOING POST-PROCESSING WHAT IS DOES IS FILTER HALLUCINATED FEATURES SO IT IS EASIER FOR GETTING ONLY ACCURATE FEATURES 
def _filter_hallucinated_features(features: list[dict]) -> list[dict]:
    """
    Three-pass hallucination filter for InternVL2 output.

    Pass 1 — boundary: remove features outside valid image area
    Pass 2 — grid pattern: detect evenly spaced coordinates
    Pass 3 — description: detect all-identical descriptions
    """
    if not features:
        return features

    # Normalize coordinates
    for f in features:
        loc = f.get("location", {})
        cx  = float(loc.get("cx", 0.5))
        cy  = float(loc.get("cy", 0.5))
        if cx > 1.0: cx = cx / 1024.0
        if cy > 1.0: cy = cy / 1024.0
        f["location"] = {"cx": round(cx, 3), "cy": round(cy, 3)}

    # Pass 1 — boundary filter
    valid = [
        f for f in features
        if f["location"]["cx"] < 0.98
        and f["location"]["cy"] < 0.98
    ]

    if len(valid) < 3:
        print(f"[Router] Features after filter: {len(valid)}/{len(features)}")
        return valid

    # Pass 2 — detect grid pattern on cx, cy, or diagonal
    cx_vals = sorted([f["location"]["cx"] for f in valid])
    cy_vals = sorted([f["location"]["cy"] for f in valid])

    def is_evenly_spaced(vals: list[float]) -> bool:
        if len(vals) < 4:
            return False
        steps = [round(vals[i+1] - vals[i], 2) for i in range(len(vals)-1)]
        unique = set(steps)
        return len(unique) == 1 and 0.08 <= list(unique)[0] <= 0.15

    cx_grid = is_evenly_spaced(cx_vals)
    cy_grid = is_evenly_spaced(cy_vals)

    # diagonal pattern: cx == cy for most features
    diagonal = sum(
        1 for f in valid
        if abs(f["location"]["cx"] - f["location"]["cy"]) < 0.05
    )
    diagonal_pattern = diagonal >= len(valid) * 0.6

    if cx_grid or cy_grid or diagonal_pattern:
        reason = []
        if cx_grid: reason.append("cx evenly spaced")
        if cy_grid: reason.append("cy evenly spaced")
        if diagonal_pattern: reason.append(f"diagonal ({diagonal}/{len(valid)})")
        print(f"[Router] Grid pattern detected ({', '.join(reason)}) — hallucination, returning []")
        return []

    # Pass 3 — detect all-same descriptions
    descriptions = [f.get("description", "").strip() for f in valid]
    non_empty    = [d for d in descriptions if d]
    if non_empty and len(set(non_empty)) == 1 and len(non_empty) >= 3:
        print(f"[Router] All {len(non_empty)} features have identical description "
              f"'{non_empty[0][:30]}' — hallucination, returning []")
        return []

    removed = len(features) - len(valid)
    if removed > 0:
        print(f"[Router] Removed {removed} boundary features")
    print(f"[Router] Features after filter: {len(valid)}/{len(features)}")
    return valid

def run_annotation_extraction(image_b64: str) -> list[dict]:
    """Extract text annotations from one tile using Qwen2.5-VL."""
    model = get_text_model()
    return model.extract_annotations(image_b64)

def run_feature_inference_from_annotations(
    annotations : list[dict],
    image_b64   : str,
) -> list[dict]:
    """
    Fallback: infer geometric features from Qwen2.5-VL annotations
    when KimiVLM returns empty or hallucinated results.
 
    Converts annotation dicts (text, location) into feature dicts
    (type, location, description) by pattern-matching annotation text
    against known manufacturing feature keywords.
    """
    if not annotations:
        print("[Router] No annotations available for fallback inference")
        return []
 
    # Keyword → feature type mapping
    FEATURE_KEYWORDS = {
        "chamfer"   : "chamfer",
        "fillet"    : "fillet",
        "radius"    : "fillet",
        "bore"      : "hole",
        "hole"      : "hole",
        "drill"     : "hole",
        "groove"    : "groove",
        "slot"      : "slot",
        "thread"    : "thread",
        "tap"       : "thread",
        "counterbore": "counterbore",
        "countersink": "countersink",
        "pocket"    : "pocket",
        "step"      : "step",
        "shoulder"  : "step",
        "keyway"    : "keyway",
        "spline"    : "spline",
        "knurl"     : "knurl",
        "undercut"  : "undercut",
    }
 
    features = []
    for ann in annotations:
        text = ann.get("text", "").lower().strip()
        loc  = ann.get("location", {})
 
        if not text:
            continue
 
        # Match first keyword found in annotation text
        feature_type = None
        for keyword, ftype in FEATURE_KEYWORDS.items():
            if keyword in text:
                feature_type = ftype
                break
 
        if feature_type is None:
            continue  # skip annotations that don't match any feature type
 
        features.append({
            "type"       : feature_type,
            "location"   : {
                "cx": loc.get("cx", 0.5),
                "cy": loc.get("cy", 0.5),
            },
            "description": ann.get("text", "").strip(),
            "source"     : "annotation_fallback",
        })
 
    print(f"[Router] Annotation fallback inferred {len(features)} features "
          f"from {len(annotations)} annotations")
    return features
 

def run_geometry_extraction_with_fallback(
    image_b64   : str,
    annotations : list[dict] = None,
) -> list[dict]:
    """
    Try InternVL2 first. If it returns empty or hallucinated results,
    fall back to inferring features from Qwen2.5-VL annotations.

    This ensures you always get some feature output even when
    InternVL2 fails on a particular tile.
    """
    features = run_geometry_extraction(image_b64)

    if not features and annotations:
        print("[Router] InternVL2 returned empty — falling back to annotation inference")
        features = run_feature_inference_from_annotations(annotations, image_b64)

    return features