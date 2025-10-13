# Pastikan sudah install: pip install paddlepaddle paddleocr

from operator import truediv
import os
import paddle
from paddle.io import Dataset, DataLoader
from paddle.optimizer import Adam
from paddle.nn import CTCLoss
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# -----------------------------
# 1. Menentukan path dataset dan label
# -----------------------------
DATASET_DIR = '/Users/user/Documents/Christopher/Project/AI/licence_detection_v2/vehicle_detection/OCR/training_model_OCR/dataset'
LABEL_FILE = '/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/vehicle_detection/OCR/training_model_OCR/dataset/labels.txt'
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DICTIONARY_FILE = '/Users/user/Documents/Christopher/Project/AI/licence_detection_v2/vehicle_detection/OCR/training_model-OCR/dictionary.txt'

# -----------------------------
# 2. Membaca label dan path gambar
# -----------------------------
def read_labels(label_file):
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            path, label = line.strip().split(' ', 1)
            samples.append((path, label))
    return samples

samples = read_labels(LABEL_FILE)

# -----------------------------
# 3. Membuat custom Dataset Paddle
# -----------------------------
class OCRRecDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # convert to grayscale
        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((100, 32))
            img = np.array(img).astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)  # (1, H, W)
        return img, label

    def __len__(self):
        return len(self.samples)

# -----------------------------
# 4. Membuat vocab karakter
# -----------------------------
def build_vocab(samples):
    chars = set()
    for _, label in samples:
        chars.update(list(label))
    vocab = ['<blank>'] + sorted(list(chars))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    return vocab, char2idx, idx2char

vocab, char2idx, idx2char = build_vocab(samples)

def encode_label(label, char2idx):
    return [char2idx[c] for c in label]

# -----------------------------
# 5. Membuat DataLoader
# -----------------------------
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

dataset = OCRRecDataset(samples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# -----------------------------
# 6. Load pretrained model PaddleOCR (recognition only)
# -----------------------------
ocr_model = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    text_recognition_model_dir=None,  # None: use default pretrained
    use_gpu=False
)

model = ocr_model.rec_model
model.train()

# -----------------------------
# 7. Loss dan Optimizer
# -----------------------------
criterion = CTCLoss()
optimizer = Adam(parameters=model.parameters(), learning_rate=LEARNING_RATE)

# -----------------------------
# 8. Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_id, (imgs, labels) in enumerate(dataloader):
        imgs = paddle.to_tensor(imgs)
        # Encode labels
        label_encoded = [encode_label(l, char2idx) for l in labels]
        label_lens = [len(l) for l in label_encoded]
        label_concat = np.concatenate([np.array(l, dtype='int32') for l in label_encoded])
        label_concat = paddle.to_tensor(label_concat)
        label_lens = paddle.to_tensor(label_lens)

        # Forward
        preds = model(imgs)
        preds_lens = paddle.to_tensor([preds.shape[0]] * BATCH_SIZE)

        # Loss
        loss = criterion(preds, label_concat, preds_lens, label_lens)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss += loss.numpy()[0]

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # -----------------------------
    # 9. Menyimpan model hasil training ke folder output/
    # -----------------------------
    paddle.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"rec_model_epoch{epoch+1}.pdparams"))

print("Training selesai. Model disimpan di folder output/.")

# -----------------------------
# Catatan:
# - Path dataset dan label sudah ditentukan di atas.
# - Konfigurasi dasar: batch size, epoch, learning rate bisa diubah sesuai kebutuhan.
# - Model hasil training akan disimpan setiap epoch di folder output/.
# - Skrip ini bisa langsung dijalankan dengan Python.
# -----------------------------