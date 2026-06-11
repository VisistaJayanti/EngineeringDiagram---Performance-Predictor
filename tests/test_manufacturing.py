import sys
import os
import io
import base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from ingestion.pdf_processor import process_upload
from ingestion.image_processor import prepare_image
from vlm.router import (
    run_annotation_extraction,
    run_geometry_extraction_with_fallback,
    get_text_model,
)
from alignment.spatial_aligner import align
from alignment.symbol_parser import parse_all
from manufacturing.process_classifier import classify_all
from manufacturing.time_estimator import refine_times

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load and prepare drawing ──────────────────────────────────────────────────
with open(os.path.join(TEST_DIR, "png2pdf.pdf"), "rb") as f:
    file_bytes = f.read()

pages     = process_upload(file_bytes, "png2pdf.pdf")
processed = prepare_image(pages[0])

print(f"Total tiles: {len(processed['tiles'])}")

# ── Run VLMs on all tiles ─────────────────────────────────────────────────────
all_features    = []
all_annotations = []

print("\nStage 1 — Running VLMs on all tiles...")
for i, tile in enumerate(processed["tiles"]):
    print(f"  Tile {i+1}/{len(processed['tiles'])}: {tile.tile_id}")

    # always extract annotations from every tile
    tile_annotations = run_annotation_extraction(tile.image_b64)

    # only run geometry extraction on top row tiles
    # bottom row tiles (tile_1_x) are title block and dimension regions
    is_geometry_tile = tile.tile_id.startswith("tile_0_")

    if is_geometry_tile:
        tile_features = run_geometry_extraction_with_fallback(
            image_b64   = tile.image_b64,
            annotations = tile_annotations,
        )
    else:
        tile_features = []
        print(f"    Skipping geometry on {tile.tile_id} (title block region)")

    # prefix IDs with tile name to avoid collisions across tiles
    for f in tile_features:
        f["feature_id"] = f"{tile.tile_id}_{f['feature_id']}"
    for a in tile_annotations:
        a["annotation_id"] = f"{tile.tile_id}_{a['annotation_id']}"

    all_features.extend(tile_features)
    all_annotations.extend(tile_annotations)

print(f"\nTotal features    : {len(all_features)}")
print(f"Total annotations : {len(all_annotations)}")

# ── VLM caller — handles both image and text-only calls ───────────────────────
text_model = get_text_model()

def vlm_caller(system_prompt, user_prompt, image_b64):
    if not image_b64:
        blank = Image.new("RGB", (32, 32), color=(255, 255, 255))
        buf   = io.BytesIO()
        blank.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return text_model.analyze(image_b64, system_prompt, user_prompt)

# ── Stage 2 — Alignment ───────────────────────────────────────────────────────
print("\nStage 2 — Aligning annotations...")
first_tile = processed["tiles"][0]
linked     = align(all_features, all_annotations, first_tile.image_b64, vlm_caller)
grounded   = parse_all(all_features, linked)

# ── Stage 3 — Manufacturing inference ────────────────────────────────────────
print("\nStage 3 — Manufacturing inference...")
report            = classify_all(grounded, vlm_caller)
report.operations = refine_times(report.operations, grounded)

print(f"\nDone.")
print(f"Total human-hours : {report.total_human_hours}")
print(f"Machines needed : {report.machines_required}")
if report.flags:
    print(f"\nFlags ({len(report.flags)}):")
    for flag in report.flags:
        print(f"  ⚠ {flag}")