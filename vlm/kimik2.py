import json
import os
import time
import openai
from dotenv import load_dotenv

from vlm.base_vlm import BaseVLM
from vlm.prompts import GEOMETRY_SYSTEM, GEOMETRY_USER


load_dotenv()


class KimiVLM(BaseVLM):

    def __init__(self):
        api_key = os.getenv("KIMI_K2_API_KEY")
        if not api_key:
            raise ValueError(
                "KIMI_K2_API_KEY not set. "
                "Get your key from openrouter.ai "
                "and add it to your .env file."
            )

        self._client = openai.OpenAI(
            api_key  = api_key,
            base_url = "https://openrouter.ai/api/v1",
        )
        print("Kimi K2.6 API client initialized")

    @property
    def model_name(self) -> str:
        return "moonshotai/kimi-k2.6"

    def analyze(
        self,
        image_b64    : str,
        system_prompt: str,
        user_prompt  : str,
        max_retries  : int = 5,
        retry_delay  : float = 10.0,   # seconds between retries
    ) -> str:
        """
        Send one tile to Kimi K2.6 and return raw string response.
        Retries automatically on 429 rate-limit errors.
        """
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model       = self.model_name,
                    max_tokens  = 4096,
                    temperature = 0.6,
                    top_p       = 0.95,
                    extra_body  = {
                        "chat_template_kwargs": {"thinking": False}
                    },
                    messages = [
                        {
                            "role"   : "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type"     : "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": user_prompt + "\nReturn JSON only.",
                                },
                            ],
                        },
                    ],
                )

                finish_reason = response.choices[0].finish_reason
                print(f"[Kimi] Finish reason: {finish_reason}")

                message = response.choices[0].message
                content = message.content or getattr(message, "reasoning_content", None)

                if not content:
                    print(f"[Kimi] Empty response. Finish reason: {finish_reason}")
                    print(f"[Kimi] Full response: {response}")
                    return '{"features": []}'

                return content

            except openai.RateLimitError as e:
                if attempt < max_retries:
                    print(f"[Kimi] Rate limited (attempt {attempt}/{max_retries}). "
                          f"Retrying in {retry_delay}s ...")
                    time.sleep(retry_delay)
                else:
                    print(f"[Kimi] Rate limit exceeded after {max_retries} attempts: {e}")
                    return '{"features": []}'

            except Exception as e:
                print(f"[Kimi] API call failed: {e}")
                return '{"features": []}'

        return '{"features": []}'

    def extract_features(self, image_b64: str) -> list[dict]:
        """
        Extract geometric features using geometry prompt.
        Returns parsed list of feature dicts.
        """
        raw     = self.analyze(image_b64, GEOMETRY_SYSTEM, GEOMETRY_USER)
        cleaned = raw.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data.get("features", [])
            return []
        except json.JSONDecodeError:
            print(f"[Kimi] Could not parse response: {cleaned[:200]}")
            return []