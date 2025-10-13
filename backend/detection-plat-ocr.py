import cv2
import easyocr
import numpy as np
from PIL import Image
import re

def preprocess_plate_image(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Thresholding untuk membuat gambar binary
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations untuk membersihkan noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def clean_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) < 3 or len(cleaned) > 12:
        return ""
    return cleaned

def parse_plate_result(plate_text):
    if not plate_text or len(plate_text) < 4:
        return {
            "nomor_polisi": "",
            "tahun_kendaraan": "",
            "bulan": "",
            "tahun": ""
        }
    
    tahun_raw = plate_text[-4:]
    nomor_polisi = plate_text[:-4]
    
    if len(tahun_raw) == 4:
        bulan = tahun_raw[:2]
        tahun_yy = tahun_raw[2:]
        
        tahun_full = f"20{tahun_yy}"
        
        return {
            "nomor_polisi": nomor_polisi,
            "tahun_kendaraan": tahun_raw, 
            "bulan": bulan,
            "tahun": tahun_full
        }
    else:
        return {
            "nomor_polisi": nomor_polisi,
            "tahun_kendaraan": tahun_raw,
            "bulan": "",
            "tahun": ""
        }

def detect_plate_text_multiple_methods(image_path):
    methods_results = []

    try:
        reader = easyocr.Reader(['en'], gpu=False)
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        if width < 200:
            scale_factor = 200 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        results = reader.readtext(img, 
                                detail=0, 
                                paragraph=True,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        
        if results:
            result3 = clean_plate_text(max(results, key=len))
            if result3:
                methods_results.append(result3)
                
    except Exception as e:
        print(f"[DEBUG] Method 3 error: {e}")
    
    if not methods_results:
        return parse_plate_result("") 
    
    from collections import Counter
    counter = Counter(methods_results)
    most_common = counter.most_common(1)
    
    if most_common[0][1] > 1: 
        final_result = most_common[0][0]
    else:
        final_result = max(methods_results, key=len)
    
    # Parse hasil final menjadi object
    parsed_result = parse_plate_result(final_result)

    
    return parsed_result

# Contoh penggunaan
if __name__ == "__main__":
    # Test dengan gambar plat
    image_path = "/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/backend/img.png"
    
    result_advanced = detect_plate_text_multiple_methods(image_path)
    print(f"\nObject Lengkap: {result_advanced}")