import os
from ultralytics import YOLO
from PIL import Image

# Paths
MODEL_PATH = '/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/vehicle-detection/model/model_v2.pt'
INPUT_FOLDER = '/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/vehicle-detection/OCR/training-model-OCR/dataset-mentah/'
OUTPUT_FOLDER = '/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/vehicle-detection/OCR/training-model-OCR/dataset/'

# Index class "plat" (0-based index, jadi 4)
PLAT_CLASS_INDEX = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith('.jpg'):
        img_path = os.path.join(INPUT_FOLDER, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        results = model(img_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
        class_ids = results[0].boxes.cls.cpu().numpy()  # class indices

        # Filter hanya class "plat"
        plat_boxes = [box for box, cls_id in zip(boxes, class_ids) if int(cls_id) == PLAT_CLASS_INDEX]

        if len(plat_boxes) == 0:
            print(f"No plate detected in {filename}.")
            continue

        for idx, bbox in enumerate(plat_boxes):
            try:
                x_min, y_min, x_max, y_max = map(int, bbox)
                cropped = image.crop((x_min, y_min, x_max, y_max))
                save_name = f"{os.path.splitext(filename)[0]}_plate{idx+1}.jpg"
                save_path = os.path.join(OUTPUT_FOLDER, save_name)
                cropped.save(save_path)
                print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error cropping/saving {filename} plate {idx+1}: {e}")

print("Cropping complete.")