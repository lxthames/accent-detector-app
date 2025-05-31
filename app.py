import os
import time
import requests
import tempfile
import streamlit as st
from urllib.parse import urlparse
from typing import Optional
import gdown
import zipfile
import librosa
import soundfile as sf
import yt_dlp
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# =========================== CONFIG ===========================
MODEL_DRIVE_ID = "19uA2hRO3aWUheXsQxXrda38QjKMCTiW1"
MODEL_ZIP_NAME = "model.zip"
MODEL_DIR = "./local_model"
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
        'audio_format': 'wav',
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

import os
import warnings
import tempfile
import streamlit as st
from typing import Optional
import librosa
import soundfile as sf
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ======================= AUDIO CONFIGURATION =======================
# Configure audio backends and suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed")
os.environ["LIBROSA_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "librosa_cache")
os.environ["SOUNDFILE_ALLOWED_EXTENSIONS"] = ".wav,.flac,.ogg"

# ======================= AUDIO CONVERSION =======================
def convert_to_wav(input_path: str, output_path: str) -> str:
    """Robust audio conversion with prioritized backends"""
    try:
        # Attempt 1: Try direct soundfile load if WAV/FLAC
        if input_path.lower().endswith(('.wav', '.flac')):
            try:
                y, sr = sf.read(input_path)
                y = librosa.to_mono(y.T) if y.ndim > 1 else y
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(output_path, y, TARGET_SR)
                return output_path
            except Exception:
                pass

        # Attempt 2: Use torchaudio's native loader
        try:
            waveform, sr = torchaudio.load(input_path)
            if sr != TARGET_SR:
                waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            torchaudio.save(output_path, waveform, TARGET_SR)
            return output_path
        except Exception:
            pass

        # Attempt 3: Librosa with explicit backend selection
        try:
            y, sr = librosa.load(
                input_path,
                sr=TARGET_SR,
                mono=True,
                res_type='kaiser_fast',
                dtype='float32'
            )
            sf.write(output_path, y, TARGET_SR)
            return output_path
        except Exception as e:
            raise AudioExtractionError(f"All conversion methods failed: {str(e)}")

    except Exception as e:
        raise AudioExtractionError(f"Audio conversion error: {str(e)}")
                    
    except Exception as e:
        raise AudioExtractionError(f"Audio conversion failed: {str(e)}")
def extract_audio_to_wav(input_path: str, output_path: str) -> str:
    """Universal audio extraction that handles both local and remote files"""
    try:
        if input_path.startswith(('http://', 'https://')):
            # Handle online videos
            temp_video = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
            download_with_retry(input_path, temp_video, is_youtube_url(input_path))
            return convert_to_wav(temp_video, output_path)
        else:
            # Handle local files
            return convert_to_wav(input_path, output_path)
    except Exception as e:
        raise AudioExtractionError(f"Audio extraction failed: {str(e)}")

# ====================== MODEL HANDLING =======================
def download_model_from_drive():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
            gdown.download(url, MODEL_ZIP_NAME, quiet=False)
            with zipfile.ZipFile(MODEL_ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            os.remove(MODEL_ZIP_NAME)
        except Exception as e:
            raise AudioExtractionError(f"Model download failed: {str(e)}")

@st.cache_resource
def load_model():
    download_model_from_drive()
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
                        extract_audio_to_wav(video_url, output_wav)
                    else:
                        temp_input = os.path.join(tmp, uploaded_file.name)
                        with open(temp_input, "wb") as f:
                            f.write(uploaded_file.read())
                        extract_audio_to_wav(temp_input, output_wav)

                    accent, confidence = detect_accent(output_wav)
                    
                    st.success(f"✅ Detected accent: {accent} (Confidence: {confidence:.2f}%)")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
