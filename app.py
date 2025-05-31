import os
import subprocess
import tempfile
import streamlit as st
from urllib.parse import urlparse
from typing import Optional
import gdown
import zipfile
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

def install_ffmpeg():
    """Install ffmpeg if not available"""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
    except:
        try:
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
        except Exception as e:
            raise AudioExtractionError(f"Failed to install ffmpeg: {str(e)}")

def download_youtube_audio(url: str, output_path: str) -> str:
    """Download YouTube audio using yt-dlp with ffmpeg fallback"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path.replace('.wav', ''),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
            'ffmpeg_location': '/usr/bin/ffmpeg'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise AudioExtractionError(f"YouTube download failed: {str(e)}")

def convert_with_torchaudio(input_path: str, output_path: str) -> str:
    """Convert any audio to WAV using torchaudio"""
    try:
        waveform, sr = torchaudio.load(input_path)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:  # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        torchaudio.save(output_path, waveform, TARGET_SR)
        return output_path
    except Exception as e:
        raise AudioExtractionError(f"Audio conversion failed: {str(e)}")

def extract_audio(input_path: str, output_path: str) -> str:
    """Main audio extraction function"""
    try:
        if input_path.startswith(('http://', 'https://')):
            if is_youtube_url(input_path):
                return download_youtube_audio(input_path, output_path)
            else:
                temp_video = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
                with requests.get(input_path, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
                    r.raise_for_status()
                    with open(temp_video, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            f.write(chunk)
                return convert_with_torchaudio(temp_video, output_path)
        else:
            return convert_with_torchaudio(input_path, output_path)
    except Exception as e:
        raise AudioExtractionError(f"Audio extraction failed: {str(e)}")

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
