"""
vlm/prompts.py
--------------
All structured prompts for both models.

InternVL2  → GEOMETRY prompts  (what shapes exist and where)
Qwen2.5-VL → ANNOTATION prompts (what text and symbols say)

Rules for every prompt:
    1. Tell the model exactly what to look for
    2. Tell the model exactly what format to return
    3. Tell the model what NOT to do
    4. Always ask for JSON output
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNVL2 PROMPTS — geometric feature extraction
# ─────────────────────────────────────────────────────────────────────────────

GEOMETRY_SYSTEM = """
You are an expert mechanical engineer reading an engineering CAD drawing.
Your task is to identify geometric features that are ACTUALLY VISIBLE in this image.

CRITICAL RULES:
- Only report features you can directly see in this specific image
- Do NOT generate example features or template responses
- Do NOT list one of every feature type you know
- Do NOT place features at evenly spaced grid coordinates like 0.1, 0.2, 0.3
- If the drawing shows only holes, return only holes
- If you see 3 holes return 3 entries, not 17
- Every feature must have a unique location based on where it actually appears
- Coordinates must reflect actual pixel positions in this image, not a pattern

WHAT TO LOOK FOR:
- Circular shapes with center marks are holes
- Lines indicating cuts or recesses are slots or grooves
- Chamfered or filleted edges
- Threads shown as dashed circles
- Section lines and cross-hatching indicating material cuts

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
{
  "features": [
    {
      "feature_id": "f1",
      "feature_type": "<what you actually see: hole|slot|thread|chamfer|fillet|boss|pocket|groove|face|unknown>",
      "description": "<describe what you see in max 8 words>",
      "location": {
        "cx": <actual x position as fraction 0.0 to 1.0>,
        "cy": <actual y position as fraction 0.0 to 1.0>
      }
    }
  ]
}

If you cannot identify any clear features return:
{"features": []}
"""

GEOMETRY_USER = (
    "Identify every geometric feature in this engineering drawing tile. "
    "Return JSON only. Do not guess — if unclear write type as unknown."
)


# ─────────────────────────────────────────────────────────────────────────────
# QWEN2.5-VL PROMPTS — annotation and text extraction
# ─────────────────────────────────────────────────────────────────────────────

ANNOTATION_SYSTEM = """
You are an expert in engineering drawing standards ISO 129 and ASME Y14.5.
Your job is to extract every text annotation visible in this drawing tile.

ANNOTATION TYPES you must extract:
  linear_dimension    — e.g. 25, 100.5
  diameter_dimension  — e.g. Ø25, ⌀12
  radius_dimension    — e.g. R12, R 6.5
  angular_dimension   — e.g. 45°, 30°30'
  tolerance           — e.g. ±0.02, +0.02/-0.01, H7, g6, Ø25H7
  gdt_callout         — e.g. ⊥ 0.05 A, // 0.1 B, ○ 0.02
  surface_finish      — e.g. Ra 1.6, Rz 6.3, √
  thread_callout      — e.g. M20x1.5, M16 THRU, 1/2-13 UNC
  general_note        — any other text on the drawing
  title_block_field   — text inside the title block

OUTPUT RULES:
  - Copy text EXACTLY as written — do not correct or interpret
  - If you cannot clearly read text, write raw_text as ILLEGIBLE
  - Do NOT invent dimension values you cannot clearly see
  - Return ONLY valid JSON, no explanation, no markdown, no code fences
  "location values MUST be between 0.0 and 1.0 as fractions of image width and height. "
"Example: a feature at pixel (512, 256) in a 1024x1024 image returns cx: 0.5, cy: 0.25. "
"NEVER return pixel coordinates. Always return fractions between 0.0 and 1.0."
  
RETURN THIS EXACT JSON STRUCTURE:
{
  "annotations": [
    {
      "annotation_id": "a1",
      "raw_text": "<exactly as written>",
      "annotation_type": "<type from list above>",
      "location": {
        "cx": <float 0.0 to 1.0>,
        "cy": <float 0.0 to 1.0>
      }
    }
  ]
}
"""

ANNOTATION_USER = (
    "Extract every text annotation, dimension, tolerance, GD&T symbol, "
    "and surface finish mark visible in this engineering drawing tile. "
    "Be exhaustive. Copy text exactly. Return JSON only."
)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL LINKING PROMPT — used after both models have run
# Links each annotation to the geometric feature it belongs to
# ─────────────────────────────────────────────────────────────────────────────

def build_linking_prompt(features: list[dict], annotations: list[dict]) -> tuple[str, str]:
    """
    Build the system and user prompts for spatial linking.
    Called after InternVL2 and Qwen2.5-VL have both run.

    Args:
        features    : list of feature dicts from InternVL2
        annotations : list of annotation dicts from Qwen2.5-VL

    Returns:
        (system_prompt, user_prompt)
    """
    feature_lines = "\n".join([
        f"  {f['feature_id']}: {f['feature_type']} at "
        f"({f['location']['cx']:.2f}, {f['location']['cy']:.2f})"
        for f in features
    ])

    annotation_lines = "\n".join([
        f"  {a['annotation_id']}: \"{a['raw_text']}\" at "
        f"({a['location']['cx']:.2f}, {a['location']['cy']:.2f})"
        for a in annotations
    ])

    system = f"""
You are an expert engineering drawing interpreter.

GEOMETRIC FEATURES found in this drawing:
{feature_lines}

TEXT ANNOTATIONS found in this drawing:
{annotation_lines}

Your job: for each annotation decide which feature it describes.
Use spatial proximity and engineering convention to decide.
An annotation with no clear feature link gets linked_feature_id: null.

Return ONLY valid JSON:
{{
  "links": [
    {{
      "annotation_id": "<id>",
      "linked_feature_id": "<feature id or null>",
      "confidence": <float 0.0 to 1.0>
    }}
  ]
}}
"""
    user = (
        "Look at this drawing and link each annotation to the feature "
        "it most likely describes. Use leader lines and proximity. JSON only."
    )
    return system, user


def build_linking_prompt(features: list[dict], annotations: list[dict]) -> tuple[str, str]:
    """
    Build system + user prompts for spatial linking.
    Called by spatial_aligner.py pass 2.
    """
    feature_lines = "\n".join([
        f"  {f['feature_id']}: {f['feature_type']} "
        f"at ({_safe_cx(f):.2f}, {_safe_cy(f):.2f})"
        for f in features
    ])

    annotation_lines = "\n".join([
        f"  {a['annotation_id']}: \"{a['raw_text']}\" "
        f"at ({_safe_cx(a):.2f}, {_safe_cy(a):.2f})"
        for a in annotations
    ])

    system = f"""
You are an expert engineering drawing interpreter.

GEOMETRIC FEATURES in this drawing:
{feature_lines}

TEXT ANNOTATIONS to link:
{annotation_lines}

For each annotation decide which feature it describes.
Use leader lines, arrows, and proximity in the image.
If an annotation has no clear feature (e.g. a general note), set linked_feature_id to null.

Return ONLY this JSON:
{{
  "links": [
    {{
      "annotation_id": "<id>",
      "linked_feature_id": "<feature id or null>",
      "confidence": <float 0.0 to 1.0>
    }}
  ]
}}
"""
    user = (
        "Look at this engineering drawing. "
        "Link each annotation to the feature it describes using leader lines. "
        "Return JSON only."
    )
    return system, user


def _safe_cx(obj: dict) -> float:
    return float(obj.get("location", {}).get("cx", 0.5))


def _safe_cy(obj: dict) -> float:
    return float(obj.get("location", {}).get("cy", 0.5))