import sys
import os

# This tells Python: look for modules in the project root folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_processor import process_upload
from ingestion.image_processor import prepare_image

# Test with a real drawing — change filename to whatever you have
with open("tests/png2pdf.pdf", "rb") as f:
    file_bytes = f.read()

pages = process_upload(file_bytes, "tests/png2pdf.pdf")
print(f"Pages found: {len(pages)}")
print(f"Page 1 size: {pages[0]['width_px']} x {pages[0]['height_px']} px")
print(f"Page 1 DPI:  {pages[0]['dpi']}")

processed = prepare_image(pages[0])
print(f"Tiles created: {len(processed['tiles'])}")
print(f"Was tiled: {processed['was_tiled']}")
for tile in processed["tiles"]:
    print(f"  {tile.tile_id} — origin ({tile.origin_x}, {tile.origin_y}) — {tile.width}x{tile.height}px")