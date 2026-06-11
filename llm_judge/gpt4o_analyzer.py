#THIS IS THE GPT ANALYZER
#IT WILL HAVE GPT KEY 


#Importing packages
import json
import io
import os 
from openai import OpenAI
from dotenv import load_dotenv 

#Loading the environment 
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

FEATURE_PROMPT = """

You are an expert mechanical engineer analyzing an engineering CAD drawing.

Extract the following from this drawing:

1. All geometric features (holes, slots, threads, chamfers, fillets etc.)
2. All text annotations (dimensions, tolerances, GD&T symbols, notes)
3. Recommended machining operations for each feature

Return ONLY this JSON structure:

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
      "operation": "<drilling|milling|turning|boring|reaming|tapping>",
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

#Now creating the function to analyze with gpt 
def _analyze_with_gpt(image_b64: str) -> dict:
    """
    Send one drawing image to GPT-4o and return structured analysis.

    Args:
        image_b64: base64 encoded PNG of the drawing

    Returns:
        dict with features, annotations, manufacturing, total_man_hours
    """

    #Now taking try and exception 
    try:

        response = client.chat.completions.create(
            model = "gpt-4o",
            max_tokens = 4096,
            temperature = 0.0,  #not doing temperature sampling
            response_format = {"type": "json_object"},
            messages = [
                {
                    "role": "system",
                    "content": FEATURE_PROMPT,
                },

                {
                    "role" : "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url" : {
                                "url" : f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this engineering drawing completely. Return JSON only."
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        print(f"[GPT-4o] Features found: {len(data.get('features', []))}")
        print(f"[GPT-4o] Annotations found: {len(data.get('annotations', []))}")
        print(f"[GPT-4o] Operations found: {len(data.get('manufacturing', []))}")
        return data 
    
    except Exception as e:
        print(f"[GPT-4o] Analysis failed: {e}")
        return {}