#Taking all the model paths 

#Importing the packages, for settings.py always first import os 
import os 
from dotenv import load_dotenv 


load_dotenv()


#Device will be 
#To get which device os.getenv 
DEVICE = os.getenv("DEVICE", "cuda")


#Now for image ingestion, the following details 
TARGET_DPI = 300
TILE_SIZE_PX = 1024
TILE_OVERLAP_PX = 128
MAX_LONG_SIDE = 4096


#Taking the model paths 
INTERNVL2_MODEL_PATH = os.getenv("INTERNVL2_MODEL_PATH", "OpenGVLab/InternVL2-8B")
QWEN25_MODEL_PATH = os.getenv("QWEN25_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

