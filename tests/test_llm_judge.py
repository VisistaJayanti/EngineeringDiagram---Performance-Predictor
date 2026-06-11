#Importing packages 


import sys
import os
import io
import base64
import json
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
from llm_judge.gpt4o_analyzer import _analyze_with_gpt
from llm_judge.gemini_analyzer import analyze_with_gemini
from llm_judge.judge import run_judge

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def run_your_pipeline(file_bytes: bytes, filename: str) -> dict:
    """Run your full pipeline and return structured output."""
    pages     = process_upload(file_bytes, filename)
    processed = prepare_image(pages[0])

    all_features    = []
    all_annotations = []

    for tile in processed["tiles"]:
        tile_annotations = run_annotation_extraction(tile.image_b64)
        is_geometry_tile = tile.tile_id.startswith("tile_0_")

        if is_geometry_tile:
            tile_features = run_geometry_extraction_with_fallback(
                image_b64   = tile.image_b64,
                annotations = tile_annotations,
            )
        else:
            tile_features = []

        for f in tile_features:
            f["feature_id"] = f"{tile.tile_id}_{f['feature_id']}"
        for a in tile_annotations:
            a["annotation_id"] = f"{tile.tile_id}_{a['annotation_id']}"

        all_features.extend(tile_features)
        all_annotations.extend(tile_annotations)

    text_model = get_text_model()

    def vlm_caller(sp, up, ib):
        if not ib:
            blank = Image.new("RGB", (32, 32), color=(255, 255, 255))
            buf   = io.BytesIO()
            blank.save(buf, format="PNG")
            ib = base64.b64encode(buf.getvalue()).decode("utf-8")
        return text_model.analyze(ib, sp, up)

    first_tile = processed["tiles"][0]
    linked     = align(all_features, all_annotations, first_tile.image_b64, vlm_caller)
    grounded   = parse_all(all_features, linked)
    report     = classify_all(grounded, vlm_caller)
    report.operations = refine_times(report.operations, grounded)

    # convert to dict for judge comparison
    return {
        "features": [
            {
                "feature_type": f.feature_type,
                "description" : f.description,
            }
            for f in grounded
        ],
        "annotations": [
            {
                "raw_text"       : a.get("raw_text", ""),
                "annotation_type": a.get("annotation_type", ""),
            }
            for a in all_annotations
            if a.get("raw_text") and a.get("raw_text") != "ILLEGIBLE"
        ],
        "manufacturing": [
            {
                "feature_description"     : op.feature_id,
                "operation"               : op.operation_type,
                "machine"                 : op.machine_type,
                "tooling"                 : ", ".join(op.tooling),
                "precision_grade"         : op.precision_class,
                "estimated_cycle_time_min": op.cycle_time_min,
                "estimated_setup_time_min": op.setup_time_min,
            }
            for op in report.operations
        ],
        "total_human_hours": report.total_human_hours,
    }


def main():
    drawing_path = os.path.join(TEST_DIR, "png2pdf.pdf")

    with open(drawing_path, "rb") as f:
        file_bytes = f.read()

    # get full image for reference models
    pages     = process_upload(file_bytes, "png2pdf.pdf")
    processed = prepare_image(pages[0])
    full_image_b64 = processed["processed_b64"]

    # ── Run all three systems ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 1 — Running your pipeline...")
    print("="*60)
    output_yours = run_your_pipeline(file_bytes, "png2pdf.pdf")
    print(f"Your pipeline: {len(output_yours.get('features', []))} features, "
          f"{len(output_yours.get('manufacturing', []))} operations")

    print("\n" + "="*60)
    print("Step 2 — Running GPT-4o...")
    print("="*60)
    output_gpt4o = _analyze_with_gpt(full_image_b64)

    print("\n" + "="*60)
    print("Step 3 — Running Gemini...")
    print("="*60)
    output_gemini = analyze_with_gemini(full_image_b64)

    # ── Run judge ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 4 — Running LLM Judge (GPT-4o)...")
    print("="*60)
    evaluation = run_judge(
        output_yours  = output_yours,
        output_gpt4o  = output_gpt4o,
        output_gemini = output_gemini,
        image_b64     = full_image_b64,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "your_pipeline" : output_yours,
        "gpt4o"         : output_gpt4o,
        "gemini"        : output_gemini,
        "evaluation"    : evaluation,
    }

    output_path = os.path.join(TEST_DIR, "judge_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()