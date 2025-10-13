import os
import streamlit as st
from PIL import Image

# --- Configuration ---
IMAGE_FOLDER = '/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/vehicle-detection/OCR/training-model-OCR/dataset/'
LABEL_FILE = '/Users/user/Documents/Christopher/Project/AI/licence-detection-v2/vehicle-detection/OCR/training-model-OCR/labels.txt'

# --- Helper Functions ---
def get_image_list(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def load_image(image_path):
    return Image.open(image_path)

def save_label(image_path, label, label_file):
    try:
        with open(label_file, 'a', encoding='utf-8') as f:
            f.write(f"{image_path} {label}\n")
        return True
    except Exception as e:
        st.error(f"Error saving label: {e}")
        return False

def get_next_index(current, total):
    return (current + 1) if current + 1 < total else 0

# --- Streamlit UI ---
st.set_page_config(page_title="License Plate Labeling", layout="wide")
st.title("License Plate Image Labeling for PaddleOCR")

# Load image list
image_files = get_image_list(IMAGE_FOLDER)
total_images = len(image_files)

if total_images == 0:
    st.warning("No images found in the specified folder.")
    st.stop()

# Session state for navigation
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'label' not in st.session_state:
    st.session_state.label = ""

current_image_file = image_files[st.session_state.idx]
current_image_path = os.path.join(IMAGE_FOLDER, current_image_file)

# Layout: Image | Input | Save Button
col1, col2 = st.columns([2, 1])
with col1:
    st.image(load_image(current_image_path), caption=current_image_file, use_column_width=True)
    st.markdown(f"**Image {st.session_state.idx + 1} of {total_images}**")

with col2:
    label_input = st.text_input("License Plate Text", value=st.session_state.label, key="label_input")
    save_next = st.button("Save & Next (Enter)", help="Save label and go to next image")

    # Keyboard shortcut: Enter to save & next
    st.markdown("""
    <script>
    const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
    if (input) {
      input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          window.parent.document.querySelector('button[kind="primary"]').click();
        }
      });
    }
    </script>
    """, unsafe_allow_html=True)

    if save_next:
        if label_input.strip() == "":
            st.warning("Label cannot be empty.")
        else:
            abs_img_path = os.path.abspath(current_image_path)
            saved = save_label(abs_img_path, label_input.strip(), LABEL_FILE)
            if saved:
                st.success(f"Saved: {abs_img_path} {label_input.strip()}")
                st.session_state.label = ""
                st.session_state.idx = get_next_index(st.session_state.idx, total_images)
                st.rerun()

# Navigation controls
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
with col_nav1:
    if st.button("Previous"):
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else total_images - 1
        st.session_state.label = ""
        st.rerun()
with col_nav2:
    st.markdown(f"**Current: {st.session_state.idx + 1}/{total_images}**")
with col_nav3:
    if st.button("Next"):
        st.session_state.idx = get_next_index(st.session_state.idx, total_images)
        st.session_state.label = ""
        st.rerun()