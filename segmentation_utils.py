import requests
import numpy as np
from PIL import Image
import io

def segment_clothes(image: Image.Image, fallback_local=True):
    """
    Human parsing: แยกส่วนเสื้อ/กางเกง
    - ถ้าออนไลน์: ใช้ HuggingFace API (LIP)
    - ถ้าออฟไลน์หรือ API fail: ใช้ heuristic (แบ่งครึ่งบน=เสื้อ, ครึ่งล่าง=กางเกง เฉพาะ pixel ที่ไม่โปร่งใส)
    คืน mask (np.ndarray) ที่ label: 5=upper-clothes, 6=pants
    """
    ########################################################
    # ========== Resize image for faster processing =========
    ########################################################
    RESIZE_SIZE = 256
    orig_size = image.size
    if max(orig_size) > RESIZE_SIZE:
        scale = RESIZE_SIZE / max(orig_size)
        new_size = (int(orig_size[0]*scale), int(orig_size[1]*scale))
        image_small = image.resize(new_size, Image.BILINEAR)
    else:
        image_small = image
    ########################################################
    # ================ Try online API ======================
    ########################################################
    API_URL = "https://hf.space/embed/akhaliq/LIP_Human_Parsing/api/predict/"
    buffered = io.BytesIO()
    image_small.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                mask_bytes = bytes(result["data"][0]["mask"]["data"])
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask_small = np.array(mask_img)
                # Resize mask back to original size
                mask = np.array(Image.fromarray(mask_small).resize(orig_size, Image.NEAREST))
                return mask
        else:
            print(f"Online API error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Online API exception: {e}")
    ########################################################
    # ========== Fallback: local heuristic (offline) =========
    ########################################################
    if fallback_local:
        arr = np.array(image_small)
        h, w = arr.shape[0], arr.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        # RGBA: use alpha > 0 as foreground
        if arr.shape[-1] == 4:
            alpha = arr[...,3]
            fg_mask = alpha > 0
        else:
            fg_mask = np.ones((h, w), dtype=bool)
        # Upper half = shirt, lower half = pants
        upper = (np.arange(h) < h//2)[:,None]
        mask[(fg_mask) & upper] = 5
        mask[(fg_mask) & (~upper)] = 6
        # If mask is all zeros (empty), fallback to all foreground
        if np.all(mask == 0):
            mask[:h//2, :] = 5
            mask[h//2:, :] = 6
        # Resize mask back to original size
        mask_out = np.array(Image.fromarray(mask).resize(orig_size, Image.NEAREST))
        return mask_out
    # If all fails, return None
    print("segment_clothes: No mask produced (offline fallback)")
    return None

def extract_part(image: Image.Image, mask: np.ndarray, part_labels):
    """
    คืนภาพเฉพาะส่วนที่ label อยู่ใน part_labels (list of int)
    """
    arr = np.array(image)
    part_mask = np.isin(mask, part_labels)
    if arr.shape[-1] == 3:
        arr_rgba = np.concatenate([arr, np.full(arr.shape[:2]+(1,), 255, dtype=np.uint8)], axis=-1)
    else:
        arr_rgba = arr.copy()
    arr_rgba[...,3] = np.where(part_mask, 255, 0)
    return Image.fromarray(arr_rgba, mode="RGBA")
