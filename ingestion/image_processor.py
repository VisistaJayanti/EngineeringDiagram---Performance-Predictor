import io
import math
import base64
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config.settings import TILE_SIZE_PX, TILE_OVERLAP_PX, MAX_LONG_SIDE
from utils.image_utils import pil_to_b64, b64_to_pil


@dataclass
class Tile:
    tile_id  : str
    image_b64: str
    origin_x : int
    origin_y : int
    width    : int
    height   : int


def prepare_image(page: dict) -> dict:
    img = b64_to_pil(page["image_b64"])
    img = _deskew(img)
    img = _denoise(img)
    img = _clahe(img)
    img = _cap_size(img, MAX_LONG_SIDE)
    processed_b64 = pil_to_b64(img)
    tiles     = _tile(img)
    was_tiled = len(tiles) > 1
    return {
        **page,
        "processed_b64" : processed_b64,
        "width_px"      : img.width,
        "height_px"     : img.height,
        "tiles"         : tiles,
        "was_tiled"     : was_tiled,
    }


def _deskew(img: Image.Image) -> Image.Image:
    gray  = np.array(img.convert("L"))
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=200)
    if lines is None:
        return img
    angles = []
    for line in lines[:50]:
        rho, theta = line[0]
        angle_deg  = np.degrees(theta) - 90
        if abs(angle_deg) < 5.0:
            angles.append(angle_deg)
    if not angles:
        return img
    skew_angle = float(np.median(angles))
    if abs(skew_angle) < 0.3:
        return img
    return img.rotate(
        angle     = -skew_angle,
        expand    = True,
        fillcolor = (255, 255, 255),
        resample  = Image.BICUBIC,
    )


def _denoise(img: Image.Image) -> Image.Image:
    img_np   = np.array(img)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_np,
        h                  = 10,
        hColor             = 10,
        templateWindowSize = 7,
        searchWindowSize   = 21,
    )
    return Image.fromarray(denoised)


def _clahe(img: Image.Image) -> Image.Image:
    lab        = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_enhanced)


def _cap_size(img: Image.Image, max_side: int) -> Image.Image:
    w, h      = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _tile(img: Image.Image) -> list[Tile]:
    w, h = img.size
    if w <= TILE_SIZE_PX and h <= TILE_SIZE_PX:
        return [Tile(
            tile_id   = "tile_0_0",
            image_b64 = pil_to_b64(img),
            origin_x  = 0,
            origin_y  = 0,
            width     = w,
            height    = h,
        )]
    tiles  = []
    stride = TILE_SIZE_PX - TILE_OVERLAP_PX
    n_rows = math.ceil((h - TILE_OVERLAP_PX) / stride)
    n_cols = math.ceil((w - TILE_OVERLAP_PX) / stride)
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * stride
            y0 = row * stride
            x1 = min(x0 + TILE_SIZE_PX, w)
            y1 = min(y0 + TILE_SIZE_PX, h)
            tile_img = img.crop((x0, y0, x1, y1))
            tiles.append(Tile(
                tile_id   = f"tile_{row}_{col}",
                image_b64 = pil_to_b64(tile_img),
                origin_x  = x0,
                origin_y  = y0,
                width     = x1 - x0,
                height    = y1 - y0,
            ))
    return tiles