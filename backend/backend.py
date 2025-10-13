import io, cv2, numpy as np
from tabnanny import verbose
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr
from PIL import Image 
from fastapi.middleware.cors import CORSMiddleware
from plate_ocr import detect_plate_text_multiple_methods
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = YOLO("/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/backend/model_v2.pt")
reader = easyocr.Reader(['en'])

def is_plate_inside_vehicle(plate_bbox, vehicle_bbox, threshold=0.8):
    """Check if plate is inside vehicle based on bbox overlap"""
    px1, py1, px2, py2 = plate_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox
    
    # Calculate intersection area
    ix1 = max(px1, vx1)
    iy1 = max(py1, vy1)
    ix2 = min(px2, vx2)
    iy2 = min(py2, vy2)
    
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    plate_area = (px2 - px1) * (py2 - py1)
    
    # Check if most of the plate is inside the vehicle
    overlap_ratio = intersection_area / plate_area if plate_area > 0 else 0
    return overlap_ratio >= threshold

@app.post('/detect')
async def detect(file: UploadFile = File(...)):
    # Read image from upload
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    results = model.predict(img_cv, conf=0.4, iou=0.45, verbose=False)[0]

    # Separate vehicles and plates
    vehicles = []
    plates = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        detection = {
            "class": name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "plate_text": ""
        }
        
        if name == "plat":
            # Simpan crop ke file sementara untuk OCR lanjutan
            crop = img_cv[y1:y2, x1:x2]
            
            # Buat file sementara
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, crop)
            
            try:
                # Gunakan OCR lanjutan dari plate_ocr.py
                ocr_result = detect_plate_text_multiple_methods(temp_path)
                
                # Transfer semua data dari OCR lanjutan
                detection["plate_text"] = ocr_result["nomor_polisi"]
                detection["tahun_kendaraan"] = ocr_result["tahun_kendaraan"]
                detection["bulan"] = ocr_result["bulan"]
                detection["tahun"] = ocr_result["tahun"]
                
            except Exception as e:
                print(f"OCR Error: {e}")
                detection["plate_text"] = ""
                detection["tahun_kendaraan"] = ""
                detection["bulan"] = ""
                detection["tahun"] = ""
            finally:
                # Hapus file sementara
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
            plates.append(detection)
        else:
            # Vehicle detection (motor, mobil, etc.)
            vehicles.append(detection)
    
    # Associate plates with vehicles
    final_detections = []
    used_plates = set()
    
    for vehicle in vehicles:
        vehicle_with_plate = vehicle.copy()
        
        # Find the best matching plate for this vehicle
        best_plate = None
        best_overlap = 0
        best_plate_idx = -1
        
        for i, plate in enumerate(plates):
            if i in used_plates:
                continue
                
            if is_plate_inside_vehicle(plate["bbox"], vehicle["bbox"]):
                # Calculate overlap score to find the best plate
                px1, py1, px2, py2 = plate["bbox"]
                vx1, vy1, vx2, vy2 = vehicle["bbox"]
                
                ix1 = max(px1, vx1)
                iy1 = max(py1, vy1)
                ix2 = min(px2, vx2)
                iy2 = min(py2, vy2)
                
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                plate_area = (px2 - px1) * (py2 - py1)
                overlap_ratio = intersection_area / plate_area if plate_area > 0 else 0
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_plate = plate
                    best_plate_idx = i
        
        # Assign the best plate to the vehicle
        if best_plate:
            vehicle_with_plate["plate_text"] = best_plate["plate_text"]
            vehicle_with_plate["tahun_kendaraan"] = best_plate["tahun_kendaraan"]
            vehicle_with_plate["bulan"] = best_plate["bulan"]
            vehicle_with_plate["tahun"] = best_plate["tahun"]
            used_plates.add(best_plate_idx)
        
        final_detections.append(vehicle_with_plate)
    
    # Add any unmatched plates as separate detections
    for i, plate in enumerate(plates):
        if i not in used_plates:
            final_detections.append(plate)

    return JSONResponse({
        "detections": final_detections
    })
