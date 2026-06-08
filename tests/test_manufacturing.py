import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_processor import process_upload
from ingestion.image_processor import prepare_image
from vlm.router import run_geometry_extraction, run_annotation_extraction, get_text_model
from alignment.spatial_aligner import align
from alignment.symbol_parser import parse_all
from manufacturing.process_classifier import classify_all
from manufacturing.time_estimator import refine_times

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(TEST_DIR, "png2pdf.pdf"), "rb") as f:
    file_bytes = f.read()

pages     = process_upload(file_bytes, "png2pdf.pdf")
processed = prepare_image(pages[0])

all_features = []
all_annotations = []

print(f"Processing {len(processed['tiles'])} tiles...")
for i, tile in enumerate(processed["tiles"]):
    print(f" Tile {i+1}/{len(processed['tiles'])}: {tile.tile_id}")
    tile_features = run_geometry_extraction(tile.image_b64)
    tile_annotations = run_annotation_extraction(tile.image_b64)

    for f in tile_features:
        f["feature_id"] = f"{tile.tile_id}_{f['feature_id']}"
    for a in tile_annotations:
        a["annotation_id"] = f"{tile.tile_id}_{a['annotation_id']}"
    
    all_features.extend(tile_features)
    all_annotations.extend(tile_annotations)

print(f"Total features: {len(all_features)}")
print(f"Total annotations: {len(all_annotations)}")

tile = processed["tiles"][0]
features = all_features
annotations = all_annotations

print("Stage 2 — Aligning annotations...")
text_model  = get_text_model()

def vlm_caller(system_prompt, user_prompt, image_b64):
    # Manufacturing inference calls pass empty image_b64
    # In that case use a text-only call via analyze with a blank image
    if not image_b64:
        # create a tiny blank white image for text-only calls
        from PIL import Image
        import io, base64
        blank = Image.new("RGB", (32, 32), color=(255, 255, 255))
        buf   = io.BytesIO()
        blank.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return text_model.analyze(image_b64, system_prompt, user_prompt)

linked      = align(features, annotations, tile.image_b64, vlm_caller)
grounded    = parse_all(features, linked)

print("Stage 3 — Manufacturing inference...")
report      = classify_all(grounded, vlm_caller)
report.operations = refine_times(report.operations, grounded)

print(f"\nDone.")
print(f"Total human-hours : {report.total_human_hours}")
print(f"Machines needed : {report.machines_required}")