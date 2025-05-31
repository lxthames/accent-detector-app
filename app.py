import os
import time
import requests
import tempfile
import streamlit as st
from urllib.parse import urlparse
from typing import Optional
import gdown  # Added for Google Drive download
import zipfile  # Added for extracting zipped models
import librosa
import soundfile as sf
import moviepy
from pydub import AudioSegment
import yt_dlp
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification


# =========================== CONFIG ===========================

# Google Drive model file info (replace with your actual file)
MODEL_DRIVE_ID = "19uA2hRO3aWUheXsQxXrda38QjKMCTiW1"  # Replace with your file ID
MODEL_ZIP_NAME = "model.zip"  # Name for downloaded zip file
MODEL_DIR = "./local_model"  # Local directory for model files

TARGET_SR = 16000
ALLOWED_VIDEO_FORMATS = {'.mp4', '.mov', '.mkv', '.webm'}
CHUNK_SIZE = 8192
MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 30

ID2LABEL = {
    0: "American", 1: "Australian", 2: "British", 3: "Canadian", 4: "English",
    5: "Indian", 6: "Irish", 7: "NewZealand", 8: "NorthernIrish", 9: "Scottish",
    10: "SouthAfrican", 11: "Unknown", 12: "Welsh"
}

# ======================= UTILITY FUNCTIONS =======================

class AudioExtractionError(Exception):
    pass

def is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in {
            'youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com', 'youtube-nocookie.com'
        })
    except Exception:
        return False

def get_file_extension(url: str) -> str:
    try:
        return os.path.splitext(url.split('?')[0].split('#')[0])[1].lower()
    except IndexError:
        return ''

def download_youtube_video(url: str, output_path: str) -> str:
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'retries': 3,
        'extract_audio': True,
        'audio_format': 'mp3',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def download_direct_video(url: str, output_path: str) -> str:
    ext = get_file_extension(url)
    if ext not in ALLOWED_VIDEO_FORMATS:
        raise AudioExtractionError(f"Unsupported format '{ext}'. Allowed: {ALLOWED_VIDEO_FORMATS}")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
    return output_path

def download_with_retry(url: str, output_path: str, is_youtube: bool) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            return download_youtube_video(url, output_path) if is_youtube else download_direct_video(url, output_path)
        except Exception as e:
            time.sleep(2 ** attempt)
    raise AudioExtractionError("Failed to download video after multiple retries.")

def convert_to_wav(input_path: str, output_path: str) -> str:
    # Load audio and resample to target SR
    y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    sf.write(output_path, y, TARGET_SR)
    return output_path

def extract_audio_to_wav(video_path: str, wav_output_path: str) -> str:
    # Extract audio using moviepy
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(wav_output_path, verbose=False, logger=None)
    clip.close()
    return wav_output_path

def extract_audio_from_video_url(video_url: str, wav_output_path: str, temp_dir: Optional[str] = None) -> str:
    if not wav_output_path.lower().endswith('.wav'):
        raise AudioExtractionError("Output path must have .wav extension")

    temp_video_path = os.path.join(temp_dir or ".", "temp_video.mp4")
    is_yt = is_youtube_url(video_url)
    downloaded_path = download_with_retry(video_url, temp_video_path, is_yt)
    return extract_audio_to_wav(downloaded_path, wav_output_path)

# ====================== MODEL HANDLING =======================

def download_model_from_drive():
    """Download model from Google Drive if not already present"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Download zip file from Google Drive
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        output = MODEL_ZIP_NAME
        
        try:
            gdown.download(url, output, quiet=False)
            
            # Extract zip file
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            
            # Clean up zip file
            os.remove(output)
            
            st.success("✅ Model downloaded and extracted successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            raise

@st.cache_resource
def load_model():
    # Ensure model is downloaded
    download_model_from_drive()
    
    # Check if model files exist
    required_files = ['config.json', 'preprocessor_config.json', 'pytorch_model.bin']
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files):
        raise FileNotFoundError(f"Required model files not found in {MODEL_DIR}")
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return processor, model

def detect_accent(audio_path: str):
    processor, model = load_model()
    waveform, sr = torchaudio.load(audio_path)
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

# ====================== STREAMLIT UI ============================

st.set_page_config(page_title="Accent Detection", layout="centered")
st.title("🗣️ Accent Detection from Speech")
st.markdown("Upload a video/audio file **or** enter a YouTube/public video URL.")

# Show model download status
with st.spinner("🔍 Checking for model files..."):
    try:
        download_model_from_drive()
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        st.stop()

video_url = st.text_input("🔗 Enter a video URL (YouTube, Loom, etc.):")
uploaded_file = st.file_uploader("📂 Or upload a video/audio file", type=["mp4", "mov", "mkv", "webm", "mp3", "wav"])

if st.button("🔍 Detect Accent"):
    if not video_url and not uploaded_file:
        st.warning("Please provide a URL or upload a file.")
    else:
        with st.spinner("⏳ Processing..."):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    output_wav = os.path.join(tmp, "output.wav")

                    if video_url:
                        extract_audio_from_video_url(video_url, output_wav, tmp)
                    else:
                        temp_input = os.path.join(tmp, uploaded_file.name)
                        with open(temp_input, "wb") as f:
                            f.write(uploaded_file.read())
                        extract_audio_to_wav(temp_input, output_wav)

                    accent, confidence = detect_accent(output_wav)

                st.success("✅ Accent detection completed!")
                st.markdown(f"### Accent: **{accent}**")
                st.markdown(f"**Confidence**: {confidence:.2f}%")

                # Adaptive Summary
                st.markdown("---")
                st.markdown("### 🧾 Summary")

                if accent == "Unknown":
                    st.warning("The accent could not be confidently classified. It may be due to unclear audio or an accent not represented in the training data.")
                elif confidence >= 85:
                    st.markdown(
                        f"The model is **highly confident** that the speaker's accent is **{accent}**. "
                        f"This strong confidence (above 85%) suggests a very reliable prediction."
                    )
                elif 65 <= confidence < 85:
                    st.markdown(
                        f"The model predicts the accent as **{accent}** with **moderate confidence** ({confidence:.2f}%). "
                        "This indicates that the speech patterns match known characteristics of this accent, but there could be some overlap with others."
                    )
                else:
                    st.markdown(
                        f"The model suggests the accent might be **{accent}**, but the **low confidence** ({confidence:.2f}%) means the result should be taken cautiously. "
                        "Background noise, unclear speech, or rare accents might affect prediction certainty."
                    )

            except AudioExtractionError as e:
                st.error(f"⚠️ {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
