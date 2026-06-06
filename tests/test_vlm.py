import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_processor import process_upload
from ingestion.image_processor import prepare_image
from vlm.router import run_geometry_extraction, run_annotation_extraction

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Load and prepare the drawing
with open(os.path.join(TEST_DIR, "png2pdf.pdf"), "rb") as f:
    file_bytes = f.read()

pages     = process_upload(file_bytes, "png2pdf.pdf")
processed = prepare_image(pages[0])

# Test on the first tile only
first_tile = processed["tiles"][0]
print(f"Testing on tile: {first_tile.tile_id}")
print(f"Tile size: {first_tile.width} x {first_tile.height}")

# Run InternVL2 — geometry extraction
print("\n--- InternVL2: Geometry Extraction ---")
features = run_geometry_extraction(first_tile.image_b64)
print(f"Features found: {len(features)}")
for f in features:
    print(f"  {f['feature_id']}: {f['feature_type']} — {f['description']}")

# Run Qwen2.5-VL — annotation extraction
print("\n--- Qwen2.5-VL: Annotation Extraction ---")
annotations = run_annotation_extraction(first_tile.image_b64)
print(f"Annotations found: {len(annotations)}")
for a in annotations:
    print(f"  {a['annotation_id']}: [{a['annotation_type']}] {a['raw_text']}")