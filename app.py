import streamlit as st
import numpy as np
from PIL import Image, ImageColor
import requests
from segmentation_utils import segment_clothes, extract_part

from sklearn.cluster import KMeans
import webcolors
try:
    from backgroundremover import remove as remove_bg_sota
    BGREMOVER_AVAILABLE = True
except ImportError:
    BGREMOVER_AVAILABLE = False
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import io
# Utility: Convert PIL Image to base64 for HTML display
import base64
from io import BytesIO
def Image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ตารางสีมาตรฐาน (ไทย, อังกฤษ, RGB, HEX)
from color_table_th import COLOR_TABLE_TH

########################################################
# =============== Color Analysis Functions ==============
########################################################
def get_dominant_colors(image, k=10):
    """
    คืน dominant color (RGB) จากภาพ โดยไม่ดูดสีพื้นหลัง (pixel โปร่งใส, ขาว, เทา)
    - ใช้ k-means เฉพาะ pixel ที่ alpha > 0 (ถ้ามี alpha)
    - กรองขาว/เทาออก
    - ถ้า pixel เหลือน้อย fallback ใช้ pixel ทั้งหมด
    """
    arr = np.array(image)
    h, w = arr.shape[0], arr.shape[1]
    # ถ้าเป็น RGBA: ใช้ alpha > 0.6*255 เป็น foreground mask (กรอง pixel โปร่งใส/ขอบ)
    if arr.shape[-1] == 4:
        alpha = arr[...,3]
        mask_fg = alpha > 153  # 0.6*255
    else:
        mask_fg = np.ones((h, w), dtype=bool)
    # ใช้ connected component เพื่อหา region ที่ใหญ่สุด (คน/เสื้อผ้า)
    if 'CV2_AVAILABLE' in globals() and CV2_AVAILABLE:
        try:
            import cv2
            mask_fg_uint8 = (mask_fg*255).astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(mask_fg_uint8)
            label_counts = np.bincount(labels_im.flatten())
            label_counts[0] = 0
            main_label = np.argmax(label_counts)
            main_mask = labels_im == main_label
            # เฉพาะจุดศูนย์กลางของ region เท่านั้น
            yx = np.argwhere(main_mask)
            if len(yx) > 0:
                cy, cx = np.median(yx, axis=0).astype(int)
                radius = max(10, int(0.18*min(h,w)))
                dist = np.sqrt((yx[:,0]-cy)**2 + (yx[:,1]-cx)**2)
                central_idx = dist < radius
                central_mask = np.zeros_like(main_mask)
                central_mask[yx[central_idx,0], yx[central_idx,1]] = True
                arr_fg = arr[...,:3][central_mask]
                # เพิ่มเติม: adaptive radius
                if len(arr_fg) < 30:
                    radius2 = max(radius*1.8, 22)
                    central_idx2 = dist < radius2
                    central_mask2 = np.zeros_like(main_mask)
                    central_mask2[yx[central_idx2,0], yx[central_idx2,1]] = True
                    arr_fg = arr[...,:3][central_mask2]
            else:
                arr_fg = arr[...,:3][main_mask]
            # กรอง pixel ที่อาจเป็นผิวหนัง (skin tone) ออก
            skin_mask = (
                (arr_fg[:,0]>90) & (arr_fg[:,0]<255) &
                (arr_fg[:,1]>40) & (arr_fg[:,1]<220) &
                (arr_fg[:,2]>30) & (arr_fg[:,2]<200) &
                (np.abs(arr_fg[:,0]-arr_fg[:,1])<55) & (np.abs(arr_fg[:,1]-arr_fg[:,2])<55)
            )
            arr_fg = arr_fg[~skin_mask]
            # เพิ่ม strict_mask: กรองขอบภาพและ pixel ผิดพลาดออก
            strict_mask = np.ones(len(arr_fg), dtype=bool)
            # กรอง pixel ที่อยู่ใกล้ขอบภาพ (เช่น x หรือ y ใกล้ 0 หรือ h/w)
            if len(arr_fg) > 0:
                # สร้าง mask จากตำแหน่ง pixel เดิม (yx)
                # เฉพาะ pixel ที่อยู่ใน central_mask หรือ main_mask
                # กำหนด threshold ขอบภาพ เช่น 8% ของขนาดภาพ
                edge_thresh = int(0.08 * min(h, w))
                # สร้างตำแหน่ง pixel ของ arr_fg
                if 'central_mask2' in locals():
                    mask_used = central_mask2
                elif 'central_mask' in locals():
                    mask_used = central_mask
                else:
                    mask_used = main_mask
                yx_fg = np.argwhere(mask_used)
                if len(yx_fg) == len(arr_fg):
                    y, x = yx_fg[:,0], yx_fg[:,1]
                    strict_mask &= (y > edge_thresh) & (y < h-edge_thresh) & (x > edge_thresh) & (x < w-edge_thresh)
                arr_fg = arr_fg[strict_mask]
        except Exception:
            arr_fg = arr[...,:3][mask_fg]
    else:
        arr_fg = arr[...,:3][mask_fg]
    # กรอง pixel ขาว/เทา/ดำเข้ม (background) แบบละเอียดขึ้น
    mask = ~(
        ((arr_fg[:,0]>220) & (arr_fg[:,1]>220) & (arr_fg[:,2]>220)) |  # ขาว
        ((np.abs(arr_fg[:,0]-arr_fg[:,1])<15) & (np.abs(arr_fg[:,1]-arr_fg[:,2])<15) & (arr_fg[:,0]>80) & (arr_fg[:,0]<210)) | # เทา
        ((arr_fg[:,0]<38) & (arr_fg[:,1]<38) & (arr_fg[:,2]<38)) |  # ดำเข้ม
        ((arr_fg[:,0]<60) & (arr_fg[:,1]<60) & (arr_fg[:,2]>80) & (arr_fg[:,2]<140)) | # น้ำเงินเข้ม
        ((arr_fg[:,0]<60) & (arr_fg[:,1]>80) & (arr_fg[:,1]<140) & (arr_fg[:,2]<60))   # เขียวเข้ม
    )
    arr_fg = arr_fg[mask]
    # กรอง outlier สีด้วย median filter (ลดผลกระทบจาก pixel noise)
    if len(arr_fg) > 0:
        med = np.median(arr_fg, axis=0)
        arr_fg = arr_fg[np.linalg.norm(arr_fg-med, axis=1)<80]
    # ถ้า pixel foreground เหลือน้อย fallback ใช้ pixel ทั้งหมด
    if len(arr_fg) < k:
        # ตรวจสอบ shape ของ arr ก่อน reshape
        arr_reshaped = None
        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                arr_reshaped = arr[...,:3].reshape(-1,3)
            elif arr.shape[-1] == 3:
                arr_reshaped = arr.reshape(-1,3)
        if arr_reshaped is not None:
            arr_fg = arr_reshaped
        else:
            # ถ้าไม่ใช่ 3 หรือ 4 channel ให้คืนค่าว่าง
            return np.array([[220,220,220]])
    # ลดจำนวน pixel เพื่อความเร็ว
    if len(arr_fg) > 20000:
        idx = np.random.choice(len(arr_fg), 20000, replace=False)
        arr_fg = arr_fg[idx]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=5).fit(arr_fg)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    colors = kmeans.cluster_centers_[sorted_idx].astype(int)
    return colors

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(tuple(rgb_tuple))
    except ValueError:
        min_diff = float('inf')
        closest_name = ''
        # รองรับ webcolors หลายเวอร์ชัน
        if hasattr(webcolors, 'CSS3_NAMES'):
            color_names = webcolors.CSS3_NAMES
        elif hasattr(webcolors, 'CSS3_NAMES_TO_HEX'):
            color_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())
        elif hasattr(webcolors, 'HTML4_NAMES_TO_HEX'):
            color_names = list(webcolors.HTML4_NAMES_TO_HEX.keys())
        elif hasattr(webcolors, 'CSS21_NAMES_TO_HEX'):
            color_names = list(webcolors.CSS21_NAMES_TO_HEX.keys())
        else:
            color_names = ['black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray']
        for name in color_names:
            try:
                r_c, g_c, b_c = webcolors.name_to_rgb(name)
                diff = np.linalg.norm(np.array([r_c, g_c, b_c]) - np.array(rgb_tuple))
                if diff < min_diff:
                    min_diff = diff
                    closest_name = name
            except Exception:
                continue
        return closest_name


########################################################
# =========== Color Matching & Style Evaluation =========
########################################################

########################################################
# =============== Style Prediction Functions ============
########################################################

def remove_background(image):
    """
    ลบพื้นหลัง: ใช้ backgroundremover (ดีที่สุด) ถ้ามี, fallback เป็น rembg, ถ้าไม่มีคืนภาพเดิม
    คืน PIL.Image RGBA
    """
    if BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove_bg_sota(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    if REMBG_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    return image.convert("RGBA")

def remove_background_bytes(image_bytes):
    """ลบพื้นหลังจาก bytes (input: bytes, output: PIL Image RGBA หรือ None) พร้อม refine ขอบให้เนียน"""
    if not REMBG_AVAILABLE:
        return None
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        output_image = remove(input_image)
        output_image = output_image.convert("RGBA")
        # refine ขอบ alpha
        output_image = refine_alpha_edges(output_image, method="morph+blur", ksize=5, blur_sigma=1.2)
        return output_image
    except Exception:
        return None

def remove_background_bytes_v2(image_bytes):
    """
    ลบพื้นหลังจาก bytes (input: bytes, output: PIL Image RGBA หรือ None)
    ใช้ rembg แบบตรงไปตรงมา (ไม่ refine, minimal logic)
    """
    if not REMBG_AVAILABLE:
        return None
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        output_image = remove(input_image)
        return output_image.convert("RGBA")
    except Exception as e:
        print("rembg (bytes) error:", e)
        return None

def remove_background_bytes_v3(image_bytes):
    """
    ลบพื้นหลังด้วย rembg แบบ bytes-in bytes-out (robust ที่สุด)
    input: bytes (PNG/JPEG/WebP), output: PIL Image RGBA หรือ None
    """
    if not REMBG_AVAILABLE:
        return None
    try:
        from rembg import remove
        output_bytes = remove(image_bytes)  # bytes in, bytes out
        image_nobg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        return image_nobg
    except Exception as e:
        print("rembg (bytes-in, bytes-out) error:", e)
        return None

def manual_remove_bg(image, bg_color, tolerance=30):
    """
    ลบพื้นหลังโดยใช้ threshold สี (bg_color: hex หรือ tuple, tolerance: int)
    คืนค่า RGBA (พื้นหลังโปร่งใส)
    """
    img = image.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[0], arr.shape[1]
    # Adaptive thresholding: คำนวณ mean/std ของขอบภาพ
    # Even less aggressive: further increase border width
    border_width = max(10, int(min(h, w)*0.22))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_pixels = arr[border_mask][...,:3]
    mean_border = np.mean(border_pixels, axis=0)
    std_border = np.std(border_pixels, axis=0)
    # Set tolerance to 50 for moderate mask
    color_dist = np.linalg.norm(arr[...,:3] - mean_border, axis=-1)
    adaptive_mask = (color_dist < (std_border.mean() + 50)) & border_mask
    try:
        import cv2
        arr_bgr = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2BGR)
        # Initial mask for GrabCut: probable background (border), probable foreground (center)
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        mask[border_mask] = cv2.GC_BGD
        center_mask = np.zeros((h, w), dtype=bool)
        # Shrink center mask even more for less aggressive cut
        center_margin = int(min(h, w)*0.14)
        center_mask[center_margin:-center_margin, center_margin:-center_margin] = True
        mask[center_mask] = cv2.GC_PR_FGD
        # Run GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(arr_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        grabcut_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        # Edge detection: หาขอบวัตถุ
        arr_gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(arr_gray, 80, 180)
        # Even lower edge dilation for softer cut
        kernel = np.ones((2,2), np.uint8)
        edge_mask_dil = cv2.dilate(edges, kernel, iterations=0) > 0
        # Combine: keep foreground, avoid cutting into edge
        final_mask = grabcut_mask & (~edge_mask_dil)
        # Morphological closing for smooth mask
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        # Even lower blur for softer feathering
        mask_blur = cv2.GaussianBlur(final_mask.astype(np.float32), (1,1), 0.3)
        arr[...,3] = (mask_blur*255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        # Fallback: original adaptive mask + edge exclusion
        try:
            import cv2
            arr_gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(arr_gray, 80, 180)
            edge_mask = edges > 0
            kernel = np.ones((5,5), np.uint8)
            edge_mask_dil = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2) > 0
            adaptive_mask2 = adaptive_mask & (~edge_mask_dil)
            adaptive_mask2 = cv2.morphologyEx(adaptive_mask2.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            arr[...,3][adaptive_mask2] = 0
            return Image.fromarray(arr)
        except Exception:
            arr[...,3][adaptive_mask] = 0
            return Image.fromarray(arr)


def advanced_predict_style(colors, image=None):
    """
    วิเคราะห์สไตล์แฟชั่นโดยใช้ตรรกะที่ซับซ้อนขึ้น เช่น
    - วิเคราะห์ distribution ของสีทั้งภาพ (ไม่ใช่แค่ 2 สี)
    - ตรวจจับความสดใส/ความหม่น/ความ contrast
    - ใช้ข้อมูล saturation, brightness, และความหลากหลายของสี
    - คืนค่าความมั่นใจ (probability) ร่วมด้วย
    """
    import colorsys
    arr = None
    if image is not None:
        # Ensure image is RGB (not RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        arr = np.array(image.resize((200,200)))
        arr = arr.reshape(-1,3)
        hsv = np.array([colorsys.rgb_to_hsv(*(pix/255.0)) for pix in arr])
        mean_sat = np.mean(hsv[:,1])
        mean_val = np.mean(hsv[:,2])
        std_val = np.std(hsv[:,2])
        color_variety = len(np.unique(arr, axis=0))
        rgb_avg = np.mean(arr, axis=0)
        # For top/bottom value/sat/hue, use main and second color
        r, g, b = colors[0]
        r2, g2, b2 = colors[1] if len(colors) > 1 else colors[0]
        val_top = max(colorsys.rgb_to_hsv(r/255, g/255, b/255)[2], 0)
        val_bottom = max(colorsys.rgb_to_hsv(r2/255, g2/255, b2/255)[2], 0)
        sat_top = max(colorsys.rgb_to_hsv(r/255, g/255, b/255)[1], 0)
        sat_bottom = max(colorsys.rgb_to_hsv(r2/255, g2/255, b2/255)[1], 0)
        hue_top = colorsys.rgb_to_hsv(r/255, g/255, b/255)[0]*360
        hue_bottom = colorsys.rgb_to_hsv(r2/255, g2/255, b2/255)[0]*360
    else:
        mean_sat = mean_val = std_val = color_variety = 0
        rgb_avg = [0,0,0]
        val_top = val_bottom = sat_top = sat_bottom = 0
        hue_top = hue_bottom = None
    return predict_style_advanced(mean_sat, mean_val, std_val, color_variety, rgb_avg, val_top, val_bottom, sat_top, sat_bottom, hue_top, hue_bottom)

# --- New advanced style prediction function ---
import random
def predict_style_advanced(mean_sat, mean_val, std_val, color_variety, rgb_avg, val_top, val_bottom, sat_top, sat_bottom, hue_top=None, hue_bottom=None):
    r, g, b = rgb_avg
    val_diff = abs(val_top - val_bottom)
    sat_diff = abs(sat_top - sat_bottom)

    # ถ้ามี hue เฉลี่ยของเสื้อ/กางเกงให้ใช้ หาค่าเฉลี่ยรวม
    if hue_top is not None and hue_bottom is not None:
        mean_hue = (hue_top + hue_bottom) / 2
        warm = mean_hue < 40 or mean_hue > 320
        cool = 160 < mean_hue < 260
    else:
        warm = cool = False

    # 🎯 เริ่มวิเคราะห์
    if mean_val > 0.8 and mean_sat < 0.25:
        return ("Minimal / Smart Casual", 95)
    if mean_sat > 0.6 and color_variety > 10000:
        return ("Pop / Colorful", 92)
    if mean_val < 0.3 and std_val < 0.1:
        return ("Street / Dark", 90)
    if mean_sat < 0.2 and std_val < 0.15:
        return ("Monochrome / Classic", 88)
    if abs(r - g) < 20 and abs(g - b) < 20 and mean_sat < 0.3:
        return ("Everyday Look / Casual", 85)
    if mean_sat > 0.5 and mean_val > 0.5:
        return ("Summer / Western", 87)
    if mean_sat < 0.3 and mean_val < 0.5:
        return ("Earth Tone / Autumn", 83)
    if color_variety < 2000:
        return ("Minimal / Simple", 80)
    # 🔥 เพิ่มสไตล์ใหม่:
    if mean_val > 0.75 and mean_sat < 0.3 and std_val < 0.2:
        return ("Pastel Soft Look", 90)
    if mean_val > 0.8 and mean_sat > 0.6 and warm:
        return ("Bright Warm / Spring Look", 91)
    if mean_val < 0.3 and color_variety < 1500:
        return ("Muted / Understated", 84)
    if val_diff > 0.5 and color_variety > 9000:
        return ("Bold Contrast / Statement", 88)
    if sat_diff > 0.4:
        return ("Experimental / Mixed", 82)
    # ✅ Default fallback
    dynamic_score = round((1 - std_val + mean_sat) * 50 + random.uniform(0, 10), 2)
    return ("Avant-Garde / Freestyle", min(dynamic_score, 100))

def refine_alpha_edges(image_rgba, method="morph+blur", ksize=3, blur_sigma=1.0):
    """
    ปรับขอบ alpha channel ให้คมขึ้น (morphological + blur)
    image_rgba: PIL Image RGBA
    method: "morph+blur" (default), "sharpen"
    คืนค่า PIL Image RGBA
    ถ้าไม่มี cv2 จะคืนภาพเดิม
    """
    if 'CV2_AVAILABLE' in globals() and CV2_AVAILABLE:
        try:
            import cv2
            arr = np.array(image_rgba)
            alpha = arr[...,3]
            kernel = np.ones((ksize,ksize), np.uint8)
            alpha_morph = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
            alpha_blur = cv2.GaussianBlur(alpha_morph, (ksize|1,ksize|1), blur_sigma)
            alpha_blur = cv2.GaussianBlur(alpha_blur, (ksize|1,ksize|1), blur_sigma)
            if method == "sharpen":
                sharp = cv2.addWeighted(alpha_blur, 1.5, cv2.GaussianBlur(alpha_blur, (0,0), 2), -0.5, 0)
                alpha_final = np.clip(sharp, 0, 255).astype(np.uint8)
            else:
                alpha_final = alpha_blur
            arr[...,3] = alpha_final
            return Image.fromarray(arr)
        except Exception:
            return image_rgba
    else:
        # ถ้าไม่มี cv2 ให้คืนภาพเดิม ไม่ error
        return image_rgba

def checkerboard_bg(img, size=8):
    """
    วาดลายหมากรุก checkerboard เป็นพื้นหลังให้ภาพโปร่งใส (PIL RGBA)
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    bg = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(0, h, size):
        for x in range(0, w, size):
            color = 220 if (x//size + y//size) % 2 == 0 else 180
            bg[y:y+size, x:x+size, :3] = color
            bg[y:y+size, x:x+size, 3] = 255
    out = arr.copy()
    alpha = arr[...,3:4]/255.0
    out = (arr[...,:3]*alpha + bg[...,:3]*(1-alpha)).astype(np.uint8)
    out = np.concatenate([out, np.full((h,w,1),255,dtype=np.uint8)], axis=-1)
    return Image.fromarray(out)

def remove_background_modnet_api(image):
    """
    ลบพื้นหลังด้วย MODNet (HuggingFace API)
    คืน PIL.Image RGBA หรือ None
    """
    API_URL = "https://hf.space/embed/andreasjansson/modnet/api/predict/"
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                from base64 import b64decode
                img_bytes = b64decode(result["data"][0].split(",")[-1])
                return Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        return None
    except Exception:
        return None



def remove_background_dynamic(image):
    """
    เลือกวิธีลบพื้นหลังที่ดีที่สุดแบบ dynamic
    - ถ้าเป็นภาพคน (ตรวจสอบเบื้องต้น) ใช้ MODNet ก่อน
    - ถ้าไม่สำเร็จ fallback backgroundremover > rembg
    - ถ้าไม่มีคืนภาพเดิม
    """
    # ลอง MODNet ก่อน
    img_modnet = remove_background_modnet_api(image)
    if img_modnet is not None:
        return img_modnet
    if BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove_bg_sota(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    if REMBG_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    return image.convert("RGBA")

########################################################
# ===================== UI Section =====================
########################################################
st.set_page_config(page_title="AI Stylist", layout="centered")
st.markdown("""
<style>
body, .main, .block-container {
    background: linear-gradient(120deg, #f7f8fa 0%, #e3e6ff 100%);
    font-family: 'Kanit', 'Prompt', 'Sarabun', 'Segoe UI', sans-serif;
    color: #23263a;
    transition: background 0.5s;
}
.main-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: #4b3aff;
    text-shadow: 0 6px 32px #e3e6ff, 0 1px 0 #23263a22;
    margin-bottom: 0.3em;
    letter-spacing: 2.5px;
    text-align: center;
    animation: fadeInDown 1s;
}
.subtitle {
    font-size: 1.35rem;
    color: #23263a;
    background: linear-gradient(90deg, #e3e6ff 60%, #f7f8fa88 100%);
    border-radius: 18px;
    box-shadow: 0 6px 32px #e3e6ff;
    padding: 1.5em 2.5em;
    margin-bottom: 2em;
    text-align: center;
    animation: fadeInUp 1.2s;
}
.stUpload {
    display: flex;
    justify-content: center;
    margin-bottom: 2em;
}
.stButton > button {
    background: linear-gradient(90deg, #4b3aff 0%, #bdbfff 100%);
    color: #23263a;
    font-size: 1.18rem;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.9em 2.5em;
    box-shadow: 0 6px 24px #e3e6ff;
    border: none;
    transition: background 0.3s, box-shadow 0.3s;
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #bdbfff 0%, #4b3aff 100%);
    box-shadow: 0 12px 40px #e3e6ff;
}
.stImage > img {
    border-radius: 24px;
    box-shadow: 0 6px 40px #e3e6ff;
    transition: box-shadow 0.3s;
}
.card {
    background: linear-gradient(120deg, #23263a 70%, #4b3aff22 100%);
    border-radius: 26px;
    box-shadow: 0 12px 40px #23263a55, 0 2px 12px #e3e6ff44;
    border: 2px solid #4b3aff33;
    padding: 2.8em 2.2em;
    margin-bottom: 2.2em;
    color: #f7f8fa !important;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 1.2s;
    background-image: linear-gradient(120deg, #404040 80%, #4b3aff22 100%);
    transition: box-shadow 0.3s, border 0.3s, background 0.5s;
}
.card::before {
    content: "";
    position: absolute;
    top: -40px; left: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, #4b3aff55 0%, #8c594d00 80%);
    z-index: 0;
}
.card b, .card span, .card h4 {
    color: #23263a !important;
    position: relative;
    z-index: 1;
    font-family: 'Prompt', 'Kanit', 'Sarabun', sans-serif;
}
.card h4 {
    font-size: 1.25em;
    font-weight: 800;
    margin-bottom: 0.7em;
    letter-spacing: 1px;
}
.card .color-block {
    border-radius: 12px;
    box-shadow: 0 2px 12px #23263a33;
    border: 2px solid #e3e6ff;
    width: 64px;
    height: 64px;
    margin-bottom: 0.5em;
    transition: box-shadow 0.3s, border 0.3s;
}
.card .color-block:hover {
    box-shadow: 0 6px 24px #4b3aff88;
    border: 2px solid #4b3aff;
}
.footer {
    text-align: center;
    color: #4b3aff;
    font-size: 1.18rem;
    margin-top: 1.5em;
    padding-bottom: 0.7em;
    letter-spacing: 1.2px;
    opacity: 0.88;
    animation: fadeIn 2s;
    font-family: 'Prompt', 'Kanit', 'Sarabun', sans-serif;
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-40px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Stylist</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">อัปโหลดรูปภาพการแต่งตัวของคุณ ระบบจะวิเคราะห์สีเสื้อผ้าและแนะนำสีที่เหมาะสม</div>', unsafe_allow_html=True)

st.markdown("""
<style>
.stFileUploader label {
    color: #111111 !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📸 เลือกรูปภาพการแต่งตัวของคุณ", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- ลบพื้นหลัง: ลอง rembg (bytes) > backgroundremover > rembg (bytes-in, bytes-out) > MODNet API > manual_remove_bg ---
    error_msgs = []
    image_nobg_bytes = None
    bg_method = ""
    # 1. rembg (bytes)
    try:
        image_nobg_bytes = remove_background_bytes_v2(image_bytes)
        if image_nobg_bytes is not None:
            bg_method = "rembg (bytes)"
    except Exception as e:
        error_msgs.append(f"rembg (bytes) error: {e}")
    # 2. backgroundremover
    if image_nobg_bytes is None and BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            image_nobg_bytes = remove_bg_sota(img_rgba).convert("RGBA")
            if image_nobg_bytes is not None:
                bg_method = "backgroundremover (fallback)"
        except Exception as e:
            error_msgs.append(f"backgroundremover error: {e}")
    # 3. rembg (bytes-in, bytes-out)
    if image_nobg_bytes is None and REMBG_AVAILABLE:
        try:
            image_nobg_bytes = remove_background_bytes_v3(image_bytes)
            if image_nobg_bytes is not None:
                bg_method = "rembg (bytes-in, bytes-out) (fallback)"
        except Exception as e:
            error_msgs.append(f"rembg (bytes-in, bytes-out) error: {e}")
    # 4. MODNet API
    if image_nobg_bytes is None:
        try:
            image_nobg_bytes = remove_background_modnet_api(image)
            if image_nobg_bytes is not None:
                bg_method = "MODNet API (fallback)"
        except Exception as e:
            error_msgs.append(f"MODNet API error: {e}")
    # 5. Cloud API (fallback)
    if image_nobg_bytes is None:
        try:
            API_URL = "https://api.backgroundremover.io/v1/remove"
            API_TOKEN = "t42QYYRVh4PuHJCaUU5i5YWW"
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            files = {"image_file": ("image.png", buffered, "image/png")}
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            response = requests.post(API_URL, files=files, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if "image_base64" in result:
                    from base64 import b64decode
                    img_bytes = b64decode(result["image_base64"])
                    image_nobg_bytes = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                    bg_method = "Cloud API (backgroundremover.io)"
            else:
                error_msgs.append(f"Cloud API error: {response.status_code} {response.text}")
        except Exception as e:
            error_msgs.append(f"Cloud API error: {e}")
    # 6. manual_remove_bg (offline heuristic)
    if image_nobg_bytes is None:
        try:
            # ใช้ manual_remove_bg: threshold ครอบคลุมทุกโทน, tolerance 50
            image_nobg_bytes = manual_remove_bg(image, bg_color=(255,255,255), tolerance=50)
            if image_nobg_bytes is not None:
                bg_method = "manual_remove_bg (offline heuristic, all bg, tol=50, dilation)"
        except Exception as e:
            error_msgs.append(f"manual_remove_bg error: {e}")
    if image_nobg_bytes is None:
        bg_method = "rembg, backgroundremover, rembg(bytes-in), MODNet, Cloud API, manual_remove_bg fail"

    # --- แสดงภาพต้นฉบับและ Preview ตัดพื้นหลังใน card เดียวกัน ---
    st.markdown('<div class="card" style="display:flex;flex-direction:row;align-items:center;justify-content:center;gap:2em;">', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;'>
        <img src='data:image/png;base64,{0}' style='border-radius:24px;border:3px solid #4b3aff;box-shadow:0 6px 32px #23263a;width:240px;max-width:90vw;object-fit:cover;' alt='ภาพต้นฉบับที่คุณอัปโหลด'/>
        <div style='margin-top:0.7em;font-size:1.08em;color:#404040;font-weight:700;'>ภาพต้นฉบับที่คุณอัปโหลด</div>
    </div>
    """.format(
        Image_to_base64(image)
    ), unsafe_allow_html=True)
    if image_nobg_bytes is not None and not (bg_method.startswith("manual_remove_bg")):
        st.image(checkerboard_bg(image_nobg_bytes.resize((120,120))), caption=f"Preview ตัดพื้นหลัง ({bg_method})", use_container_width=False, width=120)
    elif image_nobg_bytes is None:
        cloud_api_dns_error = any(
            "NameResolutionError" in msg or "Failed to resolve" in msg or "Name or service not known" in msg
            for msg in error_msgs
        )
        if cloud_api_dns_error:
            st.error("ลบพื้นหลังไม่สำเร็จทุกวิธี\nCloud API: ไม่สามารถเชื่อมต่ออินเทอร์เน็ตหรือ DNS ได้\n\nโปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ต หรือแก้ไข DNS ของ devcontainer/VM เช่น เพิ่ม nameserver 8.8.8.8 ใน /etc/resolv.conf แล้วลองใหม่อีกครั้ง\n\nรายละเอียด:\n" + "\n".join(error_msgs))
        else:
            st.error("ลบพื้นหลังไม่สำเร็จด้วยทุกวิธี\n" + "\n".join(error_msgs) if error_msgs else "rembg (bytes), backgroundremover, rembg (bytes-in), MODNet API ลบพื้นหลังไม่สำเร็จ")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Human Parsing: แยกเสื้อ/กางเกง ---
    mask = segment_clothes(image_nobg_bytes if image_nobg_bytes is not None else image)
    upper_img, lower_img = None, None
    if mask is not None:
        upper_img = extract_part(image_nobg_bytes if image_nobg_bytes is not None else image, mask, part_labels=[5])
        lower_img = extract_part(image_nobg_bytes if image_nobg_bytes is not None else image, mask, part_labels=[6])

    # --- วิเคราะห์สีและแนะนำสี (เสื้อ) ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4b3aff;font-weight:800;'>👕 สีเสื้อ (Upper Clothes)</h4>", unsafe_allow_html=True)
    if upper_img is not None:
        upper_color = get_dominant_colors(upper_img, k=3)[0]
        upper_hex = rgb_to_hex(upper_color)
        upper_rgb = f"RGB({upper_color[0]}, {upper_color[1]}, {upper_color[2]})"
        # หา closest color name/โทนสี จาก COLOR_TABLE_TH
        def closest_color_info(rgb, color_table):
            arr = np.array([c['rgb'] for c in color_table])
            dists = np.linalg.norm(arr - np.array(rgb), axis=1)
            idx = np.argmin(dists)
            return color_table[idx]
        upper_info = closest_color_info(tuple(upper_color), COLOR_TABLE_TH)
        upper_tone = upper_info.get('tone', '-')
        st.markdown(f"""
        <div style='display:flex;flex-direction:row;gap:2em;'>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{upper_hex};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>สีหลัก</div>
            <div style='font-size:0.88em;color:#111111;'>{upper_hex}</div>
            <div style='font-size:0.85em;color:#111111;'>{upper_rgb}</div>
            <div style='font-size:0.85em;color:#404040;margin-top:0.3em;'>โทน: <b>{upper_tone}</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        import colorsys
        r, g, b = upper_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        # Complementary
        comp_h = (h + 0.5) % 1.0
        comp_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(comp_h, s, v))
        comp_hex = rgb_to_hex(comp_rgb)
        comp_rgb_str = f"RGB({comp_rgb[0]}, {comp_rgb[1]}, {comp_rgb[2]})"
        # Analogous
        ana_h = (h + 0.08) % 1.0
        ana_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(ana_h, s, v))
        ana_hex = rgb_to_hex(ana_rgb)
        ana_rgb_str = f"RGB({ana_rgb[0]}, {ana_rgb[1]}, {ana_rgb[2]})"
        # Triadic
        tri_h = (h + 1/3) % 1.0
        tri_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(tri_h, s, v))
        tri_hex = rgb_to_hex(tri_rgb)
        tri_rgb_str = f"RGB({tri_rgb[0]}, {tri_rgb[1]}, {tri_rgb[2]})"
        st.markdown("<hr style='border:1px solid #404040;margin:1em 0;'>", unsafe_allow_html=True)
        st.markdown("<b>🎨 แนะนำสีตามทฤษฎี:</b>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;flex-direction:row;gap:2em;'>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{0};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>Complementary</div>
            <div style='font-size:0.88em;color:#111111;'>{1}</div>
            <div style='font-size:0.85em;color:#111111;'>{2}</div>
          </div>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{3};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>Analogous</div>
            <div style='font-size:0.88em;color:#111111;'>{4}</div>
            <div style='font-size:0.85em;color:#111111;'>{5}</div>
          </div>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{6};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>Triadic</div>
            <div style='font-size:0.88em;color:#111111;'>{7}</div>
            <div style='font-size:0.85em;color:#111111;'>{8}</div>
          </div>
        </div>
        """.format(
            comp_hex, comp_rgb_str, comp_hex,
            ana_hex, ana_rgb_str, ana_hex,
            tri_hex, tri_rgb_str, tri_hex
        ), unsafe_allow_html=True)
        # ไม่ต้องแสดงคำอธิบาย Analogous และ Triadic ตามที่ผู้ใช้ต้องการ
    else:
        st.info("ไม่พบส่วนเสื้อในภาพนี้")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- วิเคราะห์สีและแนะนำสี (กางเกง) ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4b3aff;font-weight:800;'>👖 สีกางเกง (Lower Clothes)</h4>", unsafe_allow_html=True)
    if lower_img is not None:
        lower_color = get_dominant_colors(lower_img, k=3)[0]
        lower_hex = rgb_to_hex(lower_color)
        lower_rgb = f"RGB({lower_color[0]}, {lower_color[1]}, {lower_color[2]})"
        lower_info = closest_color_info(tuple(lower_color), COLOR_TABLE_TH)
        lower_tone = lower_info.get('tone', '-')
        st.markdown(f"""
        <div style='display:flex;flex-direction:row;gap:2em;'>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{lower_hex};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>สีหลัก</div>
            <div style='font-size:0.88em;color:#111111;'>{lower_hex}</div>
            <div style='font-size:0.85em;color:#111111;'>{lower_rgb}</div>
            <div style='font-size:0.85em;color:#404040;margin-top:0.3em;'>โทน: <b>{lower_tone}</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        import colorsys
        r, g, b = lower_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        # Complementary
        comp_h = (h + 0.5) % 1.0
        comp_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(comp_h, s, v))
        comp_hex = rgb_to_hex(comp_rgb)
        comp_rgb_str = f"RGB({comp_rgb[0]}, {comp_rgb[1]}, {comp_rgb[2]})"
        # Analogous
        ana_h = (h + 0.08) % 1.0
        ana_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(ana_h, s, v))
        ana_hex = rgb_to_hex(ana_rgb)
        ana_rgb_str = f"RGB({ana_rgb[0]}, {ana_rgb[1]}, {ana_rgb[2]})"
        # Triadic
        tri_h = (h + 1/3) % 1.0
        tri_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(tri_h, s, v))
        tri_hex = rgb_to_hex(tri_rgb)
        tri_rgb_str = f"RGB({tri_rgb[0]}, {tri_rgb[1]}, {tri_rgb[2]})"
        st.markdown("<hr style='border:1px solid #404040;margin:1em 0;'>", unsafe_allow_html=True)
        st.markdown("<b>🎨 แนะนำสีตามทฤษฎี:</b>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;flex-direction:row;gap:2em;'>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{0};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>Complementary</div>
            <div style='font-size:0.88em;color:#111111;'>{1}</div>
            <div style='font-size:0.85em;color:#111111;'>{2}</div>
          </div>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{3};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>Analogous</div>
            <div style='font-size:0.88em;color:#111111;'>{4}</div>
            <div style='font-size:0.85em;color:#111111;'>{5}</div>
          </div>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{6};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>Triadic</div>
            <div style='font-size:0.88em;color:#111111;'>{7}</div>
            <div style='font-size:0.85em;color:#111111;'>{8}</div>
          </div>
        </div>
        """.format(
            comp_hex, comp_rgb_str, comp_hex,
            ana_hex, ana_rgb_str, ana_hex,
            tri_hex, tri_rgb_str, tri_hex
        ), unsafe_allow_html=True)
    else:
        st.info("ไม่พบส่วนกางเกงในภาพนี้")
    st.markdown('</div>', unsafe_allow_html=True)


# --- สรุปแนวสไตล์รวม (ไม่แยกเสื้อ/กางเกง) ---

# --- Move overall style summary card inside the uploaded_file block ---
if uploaded_file:
    # ...existing code...
    # --- สรุปแนวสไตล์รวม (ไม่แยกเสื้อ/กางเกง) ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4b3aff;font-weight:800;'>🧑‍🎤 แนวสไตล์โดยรวม (Overall Style)</h4>", unsafe_allow_html=True)

    # ใช้ภาพรวม (ไม่มีการแยกส่วน) หรือรวม upper/lower ถ้ามี
    overall_img = None
    if upper_img is not None and lower_img is not None:
        # รวมภาพ upper/lower เป็นภาพเดียว (stack vertically)
        try:
            upper_arr = np.array(upper_img)
            lower_arr = np.array(lower_img)
            # ปรับขนาดให้กว้างเท่ากัน
            min_w = min(upper_arr.shape[1], lower_arr.shape[1])
            if upper_arr.shape[1] != min_w:
                from PIL import Image
                upper_img = upper_img.resize((min_w, upper_arr.shape[0]))
                upper_arr = np.array(upper_img)
            if lower_arr.shape[1] != min_w:
                from PIL import Image
                lower_img = lower_img.resize((min_w, lower_arr.shape[0]))
                lower_arr = np.array(lower_img)
            overall_arr = np.vstack([upper_arr, lower_arr])
            overall_img = Image.fromarray(overall_arr)
        except Exception:
            overall_img = None
    elif upper_img is not None:
        overall_img = upper_img
    elif lower_img is not None:
        overall_img = lower_img
    else:
        overall_img = image

    # วิเคราะห์ dominant colors จาก overall_img
    if overall_img is not None:
        overall_colors = get_dominant_colors(overall_img, k=5)
        style_label, style_conf = advanced_predict_style(overall_colors, overall_img)
        st.markdown(f"""
        <div style='display:flex;flex-direction:row;align-items:center;gap:1.5em;'>
          <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:64px;height:64px;border-radius:10px;background:{rgb_to_hex(overall_colors[0])};border:3px solid #e3e6ff;box-shadow:0 2px 8px #23263a;'></div>
            <div style='margin-top:0.5em;font-size:0.92em;color:#111111;font-weight:600;'>สีหลัก</div>
            <div style='font-size:0.88em;color:#111111;'>{rgb_to_hex(overall_colors[0])}</div>
            <div style='font-size:0.85em;color:#111111;'>RGB({overall_colors[0][0]}, {overall_colors[0][1]}, {overall_colors[0][2]})</div>
          </div>
          <div style='margin-left:1.5em;'>
            <div style='font-size:1.15em;color:#23263a;font-weight:700;margin-bottom:0.3em;'>แนวสไตล์ที่คาดว่าเหมาะสม:</div>
            <div style='font-size:1.25em;color:#4b3aff;font-weight:900;margin-bottom:0.2em;'>{style_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ไม่สามารถวิเคราะห์แนวสไตล์โดยรวมได้")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('''<div class="footer">
<span style="font-size:1em;font-weight:600;color:#404040;">พัฒนาโดย Chanaphon Phetnoi</span><br>
<span style="font-size:0.95em;color:#404040;">รหัสนักศึกษา 664230017 | ห้อง 66/45</span><br>
<span style="font-size:0.95em;color:#404040;">นักศึกษาสาขาเทคโนโลยีสารสนเทศ</span>
</div>''', unsafe_allow_html=True)
