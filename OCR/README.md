# OCR (Optical Character Recognition) Module

Modul OCR ini merupakan bagian dari sistem deteksi dan pengenalan plat nomor kendaraan. Modul ini bertanggung jawab untuk melakukan training model OCR khusus untuk membaca teks pada plat nomor kendaraan menggunakan PaddleOCR.

## 📁 Struktur Project

```
OCR/
├── Tools/                      # Alat bantu untuk labeling dan preprocessing
├── model-OCR/                  # Model OCR yang sudah dilatih
├── training_model_OCR/         # Folder utama untuk training model OCR
└── README.md                   # Dokumentasi ini
```

## 🚀 Program Utama

**File utama dari keseluruhan sistem OCR:**
[training_model_OCR/train_paddleocr_rec_with_pipeline.py](training_model_OCR/train_paddleocr_rec_with_pipeline.py)

Program ini merupakan entry point untuk melakukan training model OCR menggunakan PaddleOCR dengan konfigurasi pipeline yang telah disesuaikan untuk pengenalan plat nomor kendaraan.

## 📂 Detail Folder dan Fungsinya

### 1. Tools/
Berisi alat bantu untuk preprocessing dan labeling data:

- **PaddleOCR-Labeler.py**: Tool untuk melakukan labeling manual pada dataset gambar plat nomor. Digunakan untuk membuat ground truth data yang diperlukan dalam proses training.

### 2. model-OCR/
Folder penyimpanan model OCR yang sudah dilatih dan siap digunakan untuk inference. Model-model di sini merupakan hasil dari proses training yang telah dilakukan.

### 3. training_model_OCR/
Folder utama yang berisi semua komponen untuk training model OCR:

#### Sub-folder dan File:
- **PaddleOCR/**: Repository lengkap PaddleOCR yang berisi framework dan tools untuk training
- **dataset/**: Dataset gambar plat nomor untuk training (berisi 150+ gambar plat nomor)
- **dataset_val/**: Dataset validasi untuk evaluasi model
- **output/**: Folder output hasil training (model, log, checkpoint)
- **venv/**: Virtual environment Python untuk isolasi dependencies

#### File Konfigurasi dan Script:
- **pipeline.yaml**: File konfigurasi utama untuk training pipeline PaddleOCR
- **dictionary.txt**: Kamus karakter yang digunakan untuk training (huruf, angka, simbol)
- **labels.txt**: File label yang berisi mapping gambar dengan teks ground truth
- **parse_labels.py**: Script untuk parsing dan preprocessing file label
- **train_paddleocr_rec.py**: Script training dasar PaddleOCR
- **train_paddleocr_rec_with_pipeline.py**: **[PROGRAM UTAMA]** Script training dengan konfigurasi pipeline
- **use-crop-plate.py**: Script untuk cropping dan preprocessing gambar plat nomor

## 🔧 Cara Penggunaan

### Prerequisites
1. Install PaddlePaddle: `pip install paddlepaddle`
2. Install PaddleOCR: `pip install paddleocr`
3. Pastikan dataset sudah tersedia di folder `dataset/`

### Menjalankan Training
```bash
cd training_model_OCR/
python3 train_paddleocr_rec_with_pipeline.py
```

### Labeling Data Baru
```bash
cd Tools/
python3 PaddleOCR-Labeler.py
```

## 📊 Dataset

Dataset terdiri dari:
- **Training Set**: 150+ gambar plat nomor kendaraan Indonesia
- **Validation Set**: Subset untuk evaluasi model
- **Format**: JPG images dengan naming convention `{id}_plate{number}.jpg`
- **Labels**: Text file berisi ground truth untuk setiap gambar

## 🎯 Tujuan Project

1. **Training Model OCR Khusus**: Melatih model PaddleOCR yang dioptimalkan untuk membaca plat nomor kendaraan Indonesia
2. **Akurasi Tinggi**: Mencapai akurasi tinggi dalam pengenalan karakter pada plat nomor
3. **Integrasi Sistem**: Model yang dihasilkan akan diintegrasikan dengan sistem deteksi kendaraan utama

## 📝 Catatan Penting

- Model OCR ini dikhususkan untuk format plat nomor kendaraan Indonesia
- Training memerlukan GPU untuk performa optimal
- Hasil training akan disimpan di folder `output/`
- Model terbaik akan dipindahkan ke folder `model-OCR/` untuk deployment

## 🔗 Dependencies

- PaddlePaddle
- PaddleOCR
- OpenCV
- NumPy
- PIL (Python Imaging Library)

---

**Developed for License Plate Detection System v2**