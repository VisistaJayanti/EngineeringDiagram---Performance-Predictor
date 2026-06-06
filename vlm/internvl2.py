"""
vlm/internvl2.py
----------------
InternVL2-8B local inference for geometric feature extraction.

What it does in this pipeline:
    Receives one tile image (base64 PNG)
    Returns JSON list of geometric features with locations

Hardware needed:
    Minimum: 1x GPU with 20GB VRAM (e.g. A100 40GB, RTX 3090)
    The 8B model in bfloat16 uses ~16GB VRAM

Download model before running:
    huggingface-cli download OpenGVLab/InternVL2-8B \
        --local-dir /your/path/InternVL2-8B

Dependencies:
    pip install transformers torch accelerate einops timm
"""

import json
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from vlm.base_vlm import BaseVLM
from vlm.prompts import GEOMETRY_SYSTEM, GEOMETRY_USER
from config.settings import INTERNVL2_MODEL_PATH, DEVICE
from utils.image_utils import b64_to_pil


# ── Image normalization constants for InternVL2 ───────────────────────────────
# These are the exact values InternVL2 was trained with — do not change
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class InternVL2(BaseVLM):

    def __init__(self):
        print(f"Loading InternVL2 from {INTERNVL2_MODEL_PATH} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            INTERNVL2_MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
        )

        self._model = AutoModel.from_pretrained(
            INTERNVL2_MODEL_PATH,
            torch_dtype=torch.bfloat16,      # half precision — saves VRAM
            low_cpu_mem_usage=True,           # load weights gradually
            trust_remote_code=True,
        ).eval().to(DEVICE)

        print(f"InternVL2 loaded on {DEVICE}")

    @property
    def model_name(self) -> str:
        return "InternVL2-8B"

    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> str:
        """
        Send one tile to InternVL2 and return raw string response.

        Steps happening inside:
            1. Decode base64 → PIL image
            2. Preprocess image to InternVL2 tensor format
            3. Build conversation with system + user prompt
            4. Run model.chat() — the InternVL2 inference interface
            5. Return raw text response
        """
        pil_img      = b64_to_pil(image_b64)
        pixel_values = self._preprocess(pil_img).to(torch.bfloat16).to(DEVICE)

        # InternVL2 uses a specific conversation format
        # System prompt is embedded inside the question
        full_question = (
            f"<system>\n{system_prompt}\n</system>\n\n"
            f"{user_prompt}\n\n"
            f"Return JSON only. No explanation."
        )

        generation_config = dict(
            max_new_tokens = 2048,
            do_sample      = False,    # deterministic — important for structured output
        )

        with torch.no_grad():
            response = self._model.chat(
                tokenizer         = self._tokenizer,
                pixel_values      = pixel_values,
                question          = full_question,
                generation_config = generation_config,
            )

        return response

    def extract_features(self, image_b64: str) -> list[dict]:
        """
        Convenience method — runs analyze with geometry prompts
        and parses the JSON response into a list of feature dicts.

        Returns empty list if model response is not valid JSON.
        """
        raw = self.analyze(image_b64, GEOMETRY_SYSTEM, GEOMETRY_USER)
        return self._parse_json(raw, key="features")

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
       
        transform = T.Compose([
            T.Resize(
                (448, 448),
                interpolation=InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(img.convert("RGB")).unsqueeze(0).to(torch.bfloat16)

    def _parse_json(self, raw: str, key: str) -> list[dict]:
        """
        Parse model response to extract the list under `key`.
        Strips markdown fences if the model adds them despite instructions.
        Returns empty list on any parse failure.
        """
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            return data.get(key, [])

        except (json.JSONDecodeError, AttributeError):
            print(f"[InternVL2] Warning: could not parse JSON response")
            print(f"[InternVL2] Raw response was: {raw[:200]}")
            return []