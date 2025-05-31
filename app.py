import os
import streamlit as st
import gdown
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ======================= CONFIG =======================
MODEL_DIR = "model"
MODEL_DRIVE_ID = "1miyNR7k89konH7_du1ORIhs8ZF8OniGv"
SAFETENSORS_FILE = "model.safetensors"

# Files that must be present in model directory beforehand
REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "tokenizer_config.json"
]

# ======================= UTILITIES =======================
def download_safetensors():
    os.makedirs(MODEL_DIR, exist_ok=True)
    safetensors_path = os.path.join(MODEL_DIR, SAFETENSORS_FILE)

    if os.path.exists(safetensors_path):
        st.info(f"✅ '{SAFETENSORS_FILE}' already exists.")
        return

    try:
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        st.write(f"⬇️ Downloading `{SAFETENSORS_FILE}` from Google Drive...")
        gdown.download(url=url, output=safetensors_path, quiet=False, fuzzy=True)

        if os.path.exists(safetensors_path):
            size = os.path.getsize(safetensors_path) / (1024 * 1024)
            st.success(f"✅ Download complete: {SAFETENSORS_FILE} ({size:.2f} MB)")
        else:
            st.error("❌ Download completed, but file not found.")

    except Exception as e:
        st.error(f"❌ Download failed: {e}")

def list_model_files():
    if not os.path.exists(MODEL_DIR):
        st.warning("📂 'model/' folder does not exist.")
        return

    files = os.listdir(MODEL_DIR)
    if not files:
        st.info("📭 No files found in 'model/' directory.")
        return

    st.write("📄 Files in `model/` directory:")
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        st.write(f"• `{f}` — {size_kb:.1f} KB")

def load_model():
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return processor, model
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return None, None

# ======================= STREAMLIT UI =======================
def main():
    st.set_page_config(page_title="Accent Model Manager", layout="centered")
    st.title("🎛️ Accent Model Manager")

    st.markdown("Use the buttons below to manage your local model setup.")

    if st.button("⬇️ Download `model.safetensors`"):
        download_safetensors()

    if st.button("📁 List files in `model/` directory"):
        list_model_files()

    if st.button("🚀 Load Model"):
        load_model()

if __name__ == "__main__":
    main()
