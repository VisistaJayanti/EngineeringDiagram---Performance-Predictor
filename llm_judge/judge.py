#Now the judge 
#This will eb the GPT-4 only 

#Using API


#Importing packahes 
import numpy as np 
import io 
from io import BytesIO 
import os 
import json 
from openai import OpenAI 
from google import genai
from PIL import Image 
from dotenv import load_dotenv 
import base64


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_judge(
    output_yours  : dict,
    output_gpt4o  : dict,
    output_gemini : dict,
    image_b64     : str,
) -> dict:
    """
    Run LLM-as-a-Judge evaluation comparing all three outputs.

    Args:
        output_yours  : your pipeline output
        output_gpt4o  : GPT-4o reference output
        output_gemini : Gemini reference output
        image_b64     : the original drawing image

    Returns:
        Evaluation report with scores and analysis
    """

    judge_prompt = f"""
You are an expert manufacturing engineer and AI evaluator.

You have three analyses of the same engineering drawing from three different systems.
Your job is to evaluate the quality of each analysis.

SYSTEM A (pipeline being evaluated):
{json.dumps(output_yours, indent=2)[:3000]}

SYSTEM B (GPT-4o reference):
{json.dumps(output_gpt4o, indent=2)[:3000]}

SYSTEM C (Gemini reference):
{json.dumps(output_gemini, indent=2)[:3000]}

Evaluate each system on these five dimensions (score 0-10 each):

1. Feature completeness: Did it find all visible features?
2. Annotation accuracy: Did it read text/dimensions correctly?
3. Feature-annotation linking: Did it correctly connect dimensions to features?
4. Manufacturing correctness: Are the process classifications correct?
5. Time estimate accuracy: Are the time estimates reasonable?

Also identify:
- Where all three systems agree (high confidence ground truth)
- Where System A differs from B and C (likely errors in pipeline)
- Specific improvements needed for System A

Return ONLY this JSON:
{{
  "scores": {{
    "system_a": {{
      "feature_completeness": <0-10>,
      "annotation_accuracy": <0-10>,
      "feature_annotation_linking": <0-10>,
      "manufacturing_correctness": <0-10>,
      "time_estimate_accuracy": <0-10>,
      "overall": <0-10>
    }},
    "system_b": {{
      "feature_completeness": <0-10>,
      "annotation_accuracy": <0-10>,
      "feature_annotation_linking": <0-10>,
      "manufacturing_correctness": <0-10>,
      "time_estimate_accuracy": <0-10>,
      "overall": <0-10>
    }},
    "system_c": {{
      "feature_completeness": <0-10>,
      "annotation_accuracy": <0-10>,
      "feature_annotation_linking": <0-10>,
      "manufacturing_correctness": <0-10>,
      "time_estimate_accuracy": <0-10>,
      "overall": <0-10>
    }}
  }},
  "agreement": [
    {{
      "topic": "<what they agree on>",
      "value": "<the agreed value>",
      "confidence": "high"
    }}
  ],
  "errors_in_system_a": [
    {{
      "category": "<feature|annotation|linking|manufacturing|time>",
      "description": "<what is wrong>",
      "system_b_says": "<what GPT-4o found>",
      "system_c_says": "<what Gemini found>",
      "severity": "<high|medium|low>"
    }}
  ],
  "improvements": [
    {{
      "priority": "<high|medium|low>",
      "suggestion": "<specific improvement>"
    }}
  ],
  "summary": "<2-3 sentence overall assessment of System A>"
}}
"""

    try:
        response = client.chat.completions.create(
            model       = "gpt-4o",
            max_tokens  = 4096,
            temperature = 0.0,
            response_format={"type": "json_object"},
            messages    = [
                {
                    "role": "system",
                    "content": "You are an expert manufacturing engineer evaluating AI systems.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url"   : f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text" : judge_prompt,
                        },
                    ],
                },
            ],
        )

        result = json.loads(response.choices[0].message.content)
        _print_evaluation(result)
        return result

    except Exception as e:
        print(f"[Judge] Evaluation failed: {e}")
        return {}


def _print_evaluation(result: dict) -> None:
    """Print evaluation results in a readable format."""
    print(f"\n{'='*60}")
    print(f"LLM-AS-A-JUDGE EVALUATION RESULTS")
    print(f"{'='*60}")

    scores = result.get("scores", {})
    for system, label in [("system_a", "Your Pipeline"),
                           ("system_b", "GPT-4o"),
                           ("system_c", "Gemini")]:
        s = scores.get(system, {})
        print(f"\n{label}:")
        print(f"  Feature completeness      : {s.get('feature_completeness', 'N/A')}/10")
        print(f"  Annotation accuracy       : {s.get('annotation_accuracy', 'N/A')}/10")
        print(f"  Feature-annotation linking: {s.get('feature_annotation_linking', 'N/A')}/10")
        print(f"  Manufacturing correctness : {s.get('manufacturing_correctness', 'N/A')}/10")
        print(f"  Time estimate accuracy    : {s.get('time_estimate_accuracy', 'N/A')}/10")
        print(f"  OVERALL                   : {s.get('overall', 'N/A')}/10")

    agreements = result.get("agreement", [])
    if agreements:
        print(f"\nHigh confidence agreements ({len(agreements)}):")
        for a in agreements[:5]:
            print(f"  ✓ {a.get('topic')}: {a.get('value')}")

    errors = result.get("errors_in_system_a", [])
    if errors:
        print(f"\nErrors found in your pipeline ({len(errors)}):")
        for e in errors:
            sev = e.get('severity', '?').upper()
            print(f"  [{sev}] {e.get('category')}: {e.get('description')}")
            print(f"         GPT-4o: {e.get('system_b_says', 'N/A')}")

    improvements = result.get("improvements", [])
    if improvements:
        print(f"\nTop improvements needed:")
        for imp in improvements[:3]:
            print(f"  [{imp.get('priority', '?').upper()}] {imp.get('suggestion')}")

    summary = result.get("summary", "")
    if summary:
        print(f"\nSummary: {summary}")

    print(f"{'='*60}")