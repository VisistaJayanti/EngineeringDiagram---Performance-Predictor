import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_processor import process_upload
from ingestion.image_processor import prepare_image
from vlm.router import run_geometry_extraction, run_annotation_extraction, get_text_model
from alignment.spatial_aligner import align
from alignment.symbol_parser import parse_all

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(TEST_DIR, "png2pdf.pdf"), "rb") as f:
    file_bytes = f.read()

pages     = process_upload(file_bytes, "png2pdf.pdf")
processed = prepare_image(pages[0])
tile      = processed["tiles"][0]

print("Running InternVL2...")
features = run_geometry_extraction(tile.image_b64)

print("Running Qwen2.5-VL...")
annotations = run_annotation_extraction(tile.image_b64)

print(f"\nFeatures   : {len(features)}")
print(f"Annotations: {len(annotations)}")

# Spatial alignment
text_model = get_text_model()

def vlm_caller(system_prompt, user_prompt, image_b64):
    return text_model.analyze(image_b64, system_prompt, user_prompt)

print("\nRunning spatial alignment...")
linked_annotations = align(
    features    = features,
    annotations = annotations,
    image_b64   = tile.image_b64,
    vlm_caller  = vlm_caller,
)

# Symbol parsing
print("\nRunning symbol parser...")
grounded_features = parse_all(
    features    = features,
    annotations = linked_annotations,
)

print(f"\nDone — {len(grounded_features)} grounded features ready for manufacturing inference")