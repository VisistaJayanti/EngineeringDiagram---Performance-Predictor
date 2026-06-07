"""
alignment/spatial_aligner.py
-----------------------------
Solves Alignment Problem 1.

Takes:
    features    : list of dicts from InternVL2
                  [{ feature_id, feature_type, description, location: {cx, cy} }]

    annotations : list of dicts from Qwen2.5-VL
                  [{ annotation_id, raw_text, annotation_type, location: {cx, cy} }]

Returns:
    Same annotations list but every dict now has an extra key:
    linked_feature_id : str or None

Two pass approach:
    Pass 1 — proximity math   : fast, no model call, handles ~70% of cases
    Pass 2 — VLM leader lines : resolves ambiguous cases using actual image
"""

import math
import json

from vlm.prompts import build_linking_prompt


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# If annotation centroid is within this distance of a feature centroid
# (in normalized 0-1 space), link them directly without a VLM call.
# 0.15 means within 15% of the image diagonal.
PROXIMITY_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def align(
    features    : list[dict],
    annotations : list[dict],
    image_b64   : str,
    vlm_caller,
) -> list[dict]:
    """
    Link every annotation to the feature it describes.

    Args:
        features    : feature dicts from InternVL2
        annotations : annotation dicts from Qwen2.5-VL
        image_b64   : base64 PNG of the drawing tile
                      (used for VLM pass 2 if needed)
        vlm_caller  : callable(system_prompt, user_prompt, image_b64) -> str
                      use Qwen2.5-VL for this — it handles text reasoning well

    Returns:
        annotations list with linked_feature_id added to every dict
    """

    # Nothing to link if either list is empty
    if not features or not annotations:
        for ann in annotations:
            ann["linked_feature_id"] = None
        return annotations

    # ── Step 1: Make a working copy so we don't mutate the originals ──────────
    annotations = [dict(ann) for ann in annotations]
    for ann in annotations:
        ann["linked_feature_id"] = None    # initialize all as unlinked

    # ── Step 2: Pass 1 — proximity linking ───────────────────────────────────
    unlinked = _proximity_pass(features, annotations)

    # ── Step 3: Pass 2 — VLM linking for anything still unlinked ─────────────
    if unlinked:
        _vlm_pass(
            unlinked_annotations = unlinked,
            features             = features,
            image_b64            = image_b64,
            vlm_caller           = vlm_caller,
            all_annotations      = annotations,
        )

    # ── Step 4: Return annotated list ─────────────────────────────────────────
    _print_summary(annotations)
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — PROXIMITY
# ─────────────────────────────────────────────────────────────────────────────

def _proximity_pass(
    features    : list[dict],
    annotations : list[dict],
) -> list[dict]:
    """
    For each annotation, find the nearest feature by Euclidean distance
    between their (cx, cy) centroids in normalized 0-1 space.

    If the nearest feature is within PROXIMITY_THRESHOLD, link them.
    Otherwise add the annotation to the unlinked list for pass 2.

    Why normalized space:
        Both models return cx, cy as fractions of image width/height
        so distance is comparable regardless of actual image pixel size.

    Returns:
        List of annotations that could NOT be linked by proximity.
    """
    unlinked = []

    for ann in annotations:
        ann_cx = _get_cx(ann)
        ann_cy = _get_cy(ann)

        best_feature_id = None
        best_distance   = float("inf")

        # Find the closest feature
        for feat in features:
            feat_cx = _get_cx(feat)
            feat_cy = _get_cy(feat)

            dist = _euclidean(ann_cx, ann_cy, feat_cx, feat_cy)

            if dist < best_distance:
                best_distance   = dist
                best_feature_id = feat["feature_id"]

        if best_distance <= PROXIMITY_THRESHOLD:
            # Close enough — link directly
            ann["linked_feature_id"] = best_feature_id
            ann["link_method"]       = "proximity"
            ann["link_confidence"]   = round(1.0 - (best_distance / PROXIMITY_THRESHOLD), 2)
        else:
            # Too far — needs VLM reasoning
            unlinked.append(ann)

    linked_count = len(annotations) - len(unlinked)
    print(f"[Aligner] Pass 1 (proximity): linked {linked_count}/{len(annotations)} annotations")
    return unlinked


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — VLM LEADER LINE REASONING
# ─────────────────────────────────────────────────────────────────────────────

def _vlm_pass(
    unlinked_annotations : list[dict],
    features             : list[dict],
    image_b64            : str,
    vlm_caller,
    all_annotations      : list[dict],
) -> None:
    """
    Send the unlinked annotations + features back to the VLM.
    The VLM looks at the actual image and uses leader lines,
    arrows, and engineering convention to resolve the links.

    Mutates all_annotations in place — sets linked_feature_id
    on any annotations that get resolved.

    Why this works:
        A dimension annotation is connected to its feature by a
        thin leader line with an arrowhead. Humans read these
        instantly. The VLM can also follow them when asked directly
        with the right prompt.
    """
    # Build the linking prompt with current features and unlinked annotations
    system_prompt, user_prompt = build_linking_prompt(
        features    = features,
        annotations = unlinked_annotations,
    )

    try:
        raw = vlm_caller(
            system_prompt = system_prompt,
            user_prompt   = user_prompt,
            image_b64     = image_b64,
        )

        # Parse the VLM response
        links = _parse_vlm_links(raw)

        # Build a lookup map for fast annotation access
        ann_map = {ann["annotation_id"]: ann for ann in all_annotations}

        # Apply the VLM's link decisions
        resolved = 0
        for link in links:
            ann_id  = link.get("annotation_id")
            feat_id = link.get("linked_feature_id")   # may be None

            if ann_id in ann_map:
                ann_map[ann_id]["linked_feature_id"] = feat_id
                ann_map[ann_id]["link_method"]       = "vlm"
                ann_map[ann_id]["link_confidence"]   = link.get("confidence", 0.5)
                if feat_id:
                    resolved += 1

        print(f"[Aligner] Pass 2 (VLM): resolved {resolved}/{len(unlinked_annotations)} remaining")

    except Exception as e:
        print(f"[Aligner] Pass 2 failed: {e} — unlinked annotations remain as None")


def _parse_vlm_links(raw: str) -> list[dict]:
    """
    Parse the VLM's linking response.
    Expected format:
    {
      "links": [
        { "annotation_id": "a1", "linked_feature_id": "f1", "confidence": 0.9 },
        { "annotation_id": "a2", "linked_feature_id": null, "confidence": 0.3 }
      ]
    }
    """
    try:
        cleaned = raw.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data.get("links", [])
        elif isinstance(data, list):
            return data
        return []

    except (json.JSONDecodeError, Exception) as e:
        print(f"[Aligner] Could not parse VLM link response: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points in normalized space."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _get_cx(obj: dict) -> float:
    """
    Extract cx from a feature or annotation dict.
    Handles both formats:
        { "location": { "cx": 0.5, "cy": 0.3 } }
    Returns 0.5 as default if missing.
    """
    loc = obj.get("location", {})
    return float(loc.get("cx", 0.5))


def _get_cy(obj: dict) -> float:
    """Extract cy from a feature or annotation dict."""
    loc = obj.get("location", {})
    return float(loc.get("cy", 0.5))


def _print_summary(annotations: list[dict]) -> None:
    """Print a clean summary of linking results."""
    linked   = [a for a in annotations if a.get("linked_feature_id")]
    unlinked = [a for a in annotations if not a.get("linked_feature_id")]

    print(f"\n[Aligner] Final summary:")
    print(f"  Total annotations : {len(annotations)}")
    print(f"  Linked            : {len(linked)}")
    print(f"  Unlinked (general): {len(unlinked)}")

    print(f"\n[Aligner] Linked pairs:")
    for ann in linked:
        print(
            f"  {ann['annotation_id']} \"{ann['raw_text']}\" "
            f"→ {ann['linked_feature_id']} "
            f"[{ann.get('link_method', '?')} "
            f"conf={ann.get('link_confidence', '?')}]"
        )

    if unlinked:
        print(f"\n[Aligner] Unlinked (no feature found):")
        for ann in unlinked:
            print(f"  {ann['annotation_id']} \"{ann['raw_text']}\"")