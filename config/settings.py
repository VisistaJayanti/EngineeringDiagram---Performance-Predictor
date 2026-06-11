import os
from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "cuda")

TARGET_DPI = int(os.getenv("TARGET_DPI", 300))
TILE_SIZE_PX = int(os.getenv("TILE_SIZE_PX", 1024))
TILE_OVERLAP_PX = int(os.getenv("TILE_OVERLAP_PX", 128))
MAX_LONG_SIDE = int(os.getenv("MAX_LONG_SIDE", 4096))

INTERNVL2_MODEL_PATH = os.getenv(
    "INTERNVL2_MODEL_PATH",
    "OpenGVLab/InternVL2-8B"
)

QWEN25_MODEL_PATH = os.getenv(
    "QWEN25_MODEL_PATH",
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

KIMI_MODEL_PATH = os.getenv(
    "KIMI_MODEL_PATH",
    "moonshotai/kimi-k2.6"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
KIMI_K2_API_KEY = os.getenv("KIMI_K2_API_KEY")