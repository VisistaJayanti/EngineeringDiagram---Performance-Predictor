#Now doing the same with gemini as well 

#Inporting packags 
import io 
import json 
import os 
from dotenv import load_dotenv
import base64 
from io import BytesIO
from PIL import Image 
import google.generativeai as genai 


load_dotenv()

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

FEATURE_PROMPT = """
You are an expert mechanical engineer analyzing an engineering CAD drawing.

Extract all geometric features, text annotations, and manufacturing operations.

Return ONLY this JSON:
{
  "features": [
    {
      "feature_type": "<type>",
      "description": "<what you see>",
      "location": {"cx": <0-1>, "cy": <0-1>}
    }
  ],
  "annotations": [
    {
      "raw_text": "<exact text>",
      "annotation_type": "<dimension|tolerance|gdt|note|surface_finish>",
      "location": {"cx": <0-1>, "cy": <0-1>}
    }
  ],
  "manufacturing": [
    {
      "feature_description": "<feature>",
      "operation": "<operation type>",
      "machine": "<machine type>",
      "tooling": "<tool required>",
      "precision_grade": "<IT grade>",
      "estimated_cycle_time_min": <float>,
      "estimated_setup_time_min": <float>
    }
  ],
  "total_man_hours": <float>,
  "confidence": <0.0 to 1.0>
}
"""


def analyze_with_gemini(image_b64: str) -> dict:
    """
    Send one drawing image to Gemini 1.5 Pro and return structured analysis.
    """
    try:
        # Convert base64 to PIL image for Gemini
        img_data = base64.b64decode(image_b64)
        pil_img  = Image.open(BytesIO(img_data)).convert("RGB")

        model    = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            [FEATURE_PROMPT, pil_img],
            generation_config=genai.types.GenerationConfig(
                temperature      = 0.0,
                max_output_tokens= 4096,
            ),
        )

        raw     = response.text.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1][4:] if parts[1].startswith("json") else parts[1]
        raw = raw.strip()

        data = json.loads(raw)
        print(f"[Gemini] Features found   : {len(data.get('features', []))}")
        print(f"[Gemini] Annotations found: {len(data.get('annotations', []))}")
        print(f"[Gemini] Operations found : {len(data.get('manufacturing', []))}")
        return data

    except Exception as e:
        print(f"[Gemini] Analysis failed: {e}")
        return {}
