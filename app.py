import os
import streamlit as st
import gdown
import torch
import torchaudio

from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ======================= CONFIG =======================
MODEL_DIR = "model"
MODEL_DRIVE_ID = "1miyNR7k89konH7_du1ORIhs8ZF8OniGv"
SAFETENSORS_FILE = "model.safetensors"
TARGET_SR = 16000

ID2LABEL = {
    0: "American", 1: "Australian", 2: "British", 3: "Canadian", 4: "English",
    5: "Indian", 6: "Irish", 7: "NewZealand", 8: "NorthernIrish", 9: "Scottish",
    10: "SouthAfrican", 11: "Unknown", 12: "Welsh"
}

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

def detect_accent(audio_bytes, processor, model):
    try:
        # Save audio temporarily
        with open("temp_input.wav", "wb") as f:
            f.write(audio_bytes)

        waveform, sr = torchaudio.load("temp_input.wav")

        # Resample
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = processor(waveform.squeeze(), sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        pred_id = torch.argmax(probs).item()
        confidence = float(probs[0, pred_id]) * 100
        label = ID2LABEL.get(pred_id, f"Label_{pred_id}")
        return label, round(confidence, 2)
    except Exception as e:
        st.error(f"❌ Error during inference: {e}")
        return None, None

# ======================= STREAMLIT UI =======================
def main():
    st.set_page_config(page_title="Accent Detection", layout="centered")
    st.title("🗣️ Accent Detection from Audio")

    # Session state initialization
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.processor = None
        st.session_state.model = None

    if st.button("⬇️ Download `model.safetensors`"):
        download_safetensors()

    if st.button("📁 List files in `model/` directory"):
        list_model_files()

    if st.button("🚀 Load Model"):
        processor, model = load_model()
        if processor and model:
            st.session_state.model_loaded = True
            st.session_state.processor = processor
            st.session_state.model = model

    st.markdown("---")
    st.markdown("### 🎧 Upload Audio File")
    audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if audio_file:
        st.audio(audio_file, format="audio/wav")

        if st.session_state.model_loaded:
            if st.button("🔍 Detect Accent"):
                with st.spinner("Processing..."):
                    label, confidence = detect_accent(
                        audio_file.read(),
                        st.session_state.processor,
                        st.session_state.model
                    )
                    if label:
                        st.success(f"### Accent: **{label}**")
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
        else:
            st.warning("⚠️ Please load the model first.")

if __name__ == "__main__":
    main()
