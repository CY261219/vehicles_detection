import os
import sys

# Pastikan PaddleOCR dan PaddlePaddle sudah terinstall
try:
    import paddle
except ImportError:
    print("PaddlePaddle belum terinstall. Install dengan: pip install paddlepaddle")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCR belum terinstall. Install dengan: pip install paddleocr")
    sys.exit(1)

def main():
    # Path ke file pipeline YAML
    pipeline_yaml = '/Users/user/Documents/Christopher/Project/AI/licence_detection_v2/vehicle_detection/OCR/training_model_OCR/pipeline.yaml'

    # Pastikan file pipeline.yaml ada
    if not os.path.exists(pipeline_yaml):
        print(f"File pipeline.yaml tidak ditemukan di: {pipeline_yaml}")
        sys.exit(1)

    # Jalankan training menggunakan tools PaddleOCR
    # Perintah ini sama seperti di terminal: python3 tools/train.py -c pipeline.yaml
    # Pastikan tools/train.py dari PaddleOCR sudah tersedia di environment Anda
    # Jika belum, clone repo PaddleOCR dan gunakan tools/train.py dari sana

    # Jika tools/train.py ada di repo lokal Anda, ganti path berikut sesuai lokasi tools/train.py
    paddleocr_tools_path = '/Users/user/Documents/Christopher/Project/AI/licence_detection_v2/vehicle_detection/OCR/training_model_OCR/PaddleOCR/tools/train.py'  # Ganti dengan path tools/train.py Anda

    # Jalankan training
    os.system(f'python3 {paddleocr_tools_path} -c {pipeline_yaml}')

if __name__ == '__main__':
    main()