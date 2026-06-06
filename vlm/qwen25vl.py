import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

from vlm.base_vlm import BaseVLM
from vlm.prompts import ANNOTATION_SYSTEM, ANNOTATION_USER
from config.settings import QWEN25_MODEL_PATH, DEVICE
from utils.image_utils import b64_to_pil


class Qwen25VL(BaseVLM):

    def __init__(self):
        print(f"Loading Qwen2.5-VL from {QWEN25_MODEL_PATH} ...")

        self._processor = AutoProcessor.from_pretrained(
            QWEN25_MODEL_PATH,
            trust_remote_code=True,
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN25_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True,
        ).eval()

        print(f"Qwen2.5-VL loaded on {DEVICE}")

    @property
    def model_name(self) -> str:
        return "Qwen2.5-VL-7B-Instruct"

    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> str:
        pil_img = b64_to_pil(image_b64)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_img,
                    },
                    {
                        "type": "text",
                        "text": user_prompt + "\nReturn JSON only. No explanation.",
                    },
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.2,
            )

        input_length = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_length:]

        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response

    def extract_annotations(self, image_b64: str) -> list[dict]:
        raw = self.analyze(image_b64, ANNOTATION_SYSTEM, ANNOTATION_USER)
        results = self._parse_json(raw, key="annotations")
        return self._deduplicate(results)

    def _parse_json(self, raw: str, key: str) -> list[dict]:
        try:
            cleaned = raw.strip()

            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            try:
                data = json.loads(cleaned)
                return data.get(key, [])

            except json.JSONDecodeError:
                print(f"[Qwen2.5-VL] JSON truncated — attempting partial recovery")

                array_start = cleaned.find('"' + key + '"')
                if array_start == -1:
                    return []

                bracket_start = cleaned.find("[", array_start)
                if bracket_start == -1:
                    return []

                partial = cleaned[bracket_start:]

                objects = []
                depth = 0
                current = ""

                for char in partial:
                    current += char
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(current.strip().strip(","))
                                objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            current = ""

                print(f"[Qwen2.5-VL] Recovered {len(objects)} objects")
                return objects

        except Exception as e:
            print(f"[Qwen2.5-VL] Parse failed: {e}")
            print(f"[Qwen2.5-VL] Raw response: {raw[:300]}")
            return []

    def _deduplicate(self, objects: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for obj in objects:
            cx = round(obj.get("location", {}).get("cx", 0), 2)
            cy = round(obj.get("location", {}).get("cy", 0), 2)
            key = (obj.get("raw_text", ""), cx, cy)
            if key not in seen:
                seen.add(key)
                unique.append(obj)
        return unique