import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_processor import process_upload
from ingestion.image_processor import prepare_image
from vlm.router import get_geometry_model

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(TEST_DIR, "png2pdf.pdf"), "rb") as f:
    file_bytes = f.read()

pages     = process_upload(file_bytes, "png2pdf.pdf")
processed = prepare_image(pages[0])

model = get_geometry_model()

# check raw output from tile_0_1 which has the most hallucinations
tile = processed["tiles"][1]
raw_features = model.extract_features(tile.image_b64)

print(f"Raw features from tile_0_1: {len(raw_features)}")
for f in raw_features:
    print(f"  {f['feature_id']}: {f['feature_type']} "
          f"loc={f.get('location')} desc={f.get('description', '')[:40]}")