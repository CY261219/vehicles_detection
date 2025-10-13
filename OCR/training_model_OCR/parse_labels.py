import os

# Ubah path ini jika file labels.txt berada di lokasi berbeda
LABELS_PATH = "dataset/labels.txt"
DELIMITER = " "  # Ubah ke "\t" jika menggunakan tab sebagai pemisah


def parse_labels(labels_path, delimiter):
    if not os.path.exists(labels_path):
        print(f"File {labels_path} tidak ditemukan.")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip("\n").strip()
        if not line:
            print(f"[Line {idx+1}] Kosong!")
            continue
        substr = line.split(delimiter)
        if len(substr) < 2:
            print(f"[Line {idx+1}] Format salah: '{line}'")
            continue
        file_name = substr[0]
        label = substr[1]
        print(f"[Line {idx+1}] Path: {file_name} | Label: {label}")

if __name__ == "__main__":
    parse_labels(LABELS_PATH, DELIMITER)