import os
import subprocess
import tempfile
import requests
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
MODEL_DRIVE_ID = "1miyNR7k89konH7_du1ORIhs8ZF8OniGv"
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

def download_youtube_audio(url: str, output_path: str) -> str:
    """Download YouTube audio using yt-dlp with FFmpeg fallback"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path.replace('.wav', ''),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
            'retries': 3
        }
        
        if os.path.exists('/usr/bin/ffmpeg'):
            ydl_opts['ffmpeg_location'] = '/usr/bin/ffmpeg'
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise AudioExtractionError(f"YouTube download failed: {str(e)}")

def convert_with_torchaudio(input_path: str, output_path: str) -> str:
    """Convert any audio file to WAV format using torchaudio"""
    try:
        waveform, sr = torchaudio.load(input_path)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        torchaudio.save(output_path, waveform, TARGET_SR)
        return output_path
    except Exception as e:
        raise AudioExtractionError(f"Audio conversion failed: {str(e)}")

def extract_audio(input_path: str, output_path: str) -> str:
    """Main audio extraction function with fallbacks"""
    try:
        if input_path.startswith(('http://', 'https://')) and is_youtube_url(input_path):
            return download_youtube_audio(input_path, output_path)
        elif input_path.startswith(('http://', 'https://')):
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

# ====================== MODEL HANDLING =======================
def download_model_from_drive():
    """Download and extract model files from Google Drive, regardless of nested folder structure"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    required_files = {
        'config.json',
        'preprocessor_config.json', 
        'pytorch_model_quantized.pt',
        'vocab.json',
        'tokenizer_config.json'
    }

    # Skip if all files already exist
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files):
        print("✅ All model files already exist in MODEL_DIR.")
        return

    try:
        # Download zip file
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_ZIP_NAME, quiet=False)

        with zipfile.ZipFile(MODEL_ZIP_NAME, 'r') as zip_ref:
            print("📦 Contents of the ZIP file:")
            for name in zip_ref.namelist():
                print(" -", name)

            found_files = set()
            for zip_info in zip_ref.infolist():
                filename = os.path.basename(zip_info.filename)
                if filename in required_files and not zip_info.is_dir():
                    # Extract to MODEL_DIR
                    target_path = os.path.join(MODEL_DIR, filename)
                    with zip_ref.open(zip_info.filename) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    print(f"✅ Extracted: {filename}")
                    found_files.add(filename)

        # Verify all required files were extracted
        missing_files = required_files - found_files
        if missing_files:
            raise AudioExtractionError(
                f"❌ Missing required files: {missing_files}\n"
                f"✅ Found files: {found_files}"
            )

    except Exception as e:
        raise AudioExtractionError(f"❌ Model setup failed: {str(e)}")

    finally:
        if os.path.exists(MODEL_ZIP_NAME):
            os.remove(MODEL_ZIP_NAME)



@st.cache_resource 
def load_model():
    """Load model with clear error reporting"""
    try:
        download_model_from_drive()
        
        # Verify quantized model exists
        model_path = os.path.join(MODEL_DIR, 'pytorch_model_quantized.pt')
        if not os.path.exists(model_path):
            available_files = "\n- ".join(os.listdir(MODEL_DIR))
            raise FileNotFoundError(
                f"Quantized model not found at {model_path}\n"
                f"Files in directory:\n- {available_files}"
            )
            
        # Load processor and config
        processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
        
        # Load quantized model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)

        model.eval()
        return processor, model
        
    except Exception as e:
        st.error("❌ Failed to load model")
        st.error(f"Error: {str(e)}")
        st.error("Please verify the zip file contains a 'model' folder with:")
        st.error("- config.json\n- preprocessor_config.json")
        st.error("- pytorch_model_quantized.pt\n- vocab.json")
        st.error("- tokenizer_config.json")
        st.stop()

def detect_accent(audio_path: str):
    """Run accent detection on audio file"""
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
def main():
    st.set_page_config(page_title="Accent Detection", layout="centered")
    st.title("🗣️ Accent Detection from Speech")
    
    # Initialize model early to catch errors
    with st.spinner("Initializing model..."):
        processor, model = load_model()

    st.markdown("Upload a video/audio file or enter a YouTube URL")

    video_url = st.text_input("🔗 Enter YouTube URL:")
    uploaded_file = st.file_uploader("📂 Or upload file", 
                                   type=["mp4", "mov", "mkv", "webm", "mp3", "wav"])

    if st.button("🔍 Detect Accent"):
        if not video_url and not uploaded_file:
            st.warning("Please provide input")
        else:
            with st.spinner("Processing..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        output_wav = os.path.join(tmp_dir, "output.wav")
                        
                        if video_url:
                            if is_youtube_url(video_url):
                                extract_audio(video_url, output_wav)
                            else:
                                raise AudioExtractionError("Only YouTube URLs supported")
                        else:
                            temp_path = os.path.join(tmp_dir, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            extract_audio(temp_path, output_wav)

                        accent, confidence = detect_accent(output_wav)
                        
                        st.success("✅ Analysis Complete")
                        st.markdown(f"### Accent: **{accent}**")
                        st.markdown(f"**Confidence**: {confidence:.2f}%")
                        
                        if confidence > 85:
                            st.info("High confidence prediction")
                        elif confidence > 60:
                            st.info("Moderate confidence prediction")
                        else:
                            st.warning("Low confidence result")

                except AudioExtractionError as e:
                    st.error(f"⚠️ Processing error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
