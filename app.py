import streamlit as st
import threading
import queue
import wave
import tempfile
import os
import time
import whisper
import requests
from fpdf import FPDF
import pyaudio
from dotenv import load_dotenv
import os

# Load Mistral API key from environment
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Audio config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SECONDS = 3
CHUNK_FRAMES = int(RATE / CHUNK * CHUNK_SECONDS)

# Initialize Whisper
model = whisper.load_model("base")

# App title
st.set_page_config(page_title="Live Transcription App", layout="wide")
st.title("🎙️ Live Audio Transcription with Mistral Correction")

# Fixed-position CSS
st.markdown("""
<style>
#fixed-buttons {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: white;
    border-top: 1px solid #ddd;
    padding: 1rem;
    z-index: 9999;
    display: flex;
    justify-content: center;
    gap: 2rem;
}
button[kind="primary"] {
    background-color: red !important;
    color: white !important;
}
#live-output {
    height: 300px;
    overflow-y: scroll;
    padding: 1rem;
    border: 1px solid #ccc;
    background: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)

# Initialize state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
    st.session_state.transcript_chunks = []
    st.session_state.raw_audio = []

audio_queue = queue.Queue()
recording_flag = threading.Event()

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    while recording_flag.is_set():
        data = stream.read(CHUNK)
        frames.append(data)
        if len(frames) >= CHUNK_FRAMES:
            chunk = b''.join(frames)
            audio_queue.put(chunk)
            st.session_state.raw_audio.append(chunk)
            frames = []
    if frames:
        chunk = b''.join(frames)
        audio_queue.put(chunk)
        st.session_state.raw_audio.append(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcribe_audio():
    while recording_flag.is_set() or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wf:
                wav_path = wf.name
                wf = wave.open(wav_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(chunk)
                wf.close()
                result = model.transcribe(wav_path, fp16=False)
                st.session_state.transcript_chunks.append(result['text'])
                os.unlink(wav_path)
        except queue.Empty:
            continue

# Output display
live_output = st.empty()

with st.container():
    full_text = " ".join(st.session_state.transcript_chunks)
    live_output.markdown(f'<div id="live-output">{full_text}</div>', unsafe_allow_html=True)

# Button controls
with st.container():
    st.markdown('<div id="fixed-buttons">', unsafe_allow_html=True)

    if not st.session_state.is_recording:
        if st.button("🔴 Transcribe", key="start"):
            st.session_state.is_recording = True
            recording_flag.set()
            threading.Thread(target=record_audio, daemon=True).start()
            threading.Thread(target=transcribe_audio, daemon=True).start()
    else:
        if st.button("⏹️ Stop", key="stop"):
            st.session_state.is_recording = False
            recording_flag.clear()
            time.sleep(2)

    # PDF download button (enabled only after Stop)
    if not st.session_state.is_recording and st.session_state.transcript_chunks:
        if st.button("📄 Download Transcript as PDF", key="download"):
            raw_text = " ".join(st.session_state.transcript_chunks)

            # Send to Mistral for cleanup
            def correct_text_with_mistral(text):
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that fixes transcription errors and grammar."},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.3
                }
                response = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            try:
                corrected_text = correct_text_with_mistral(raw_text)
            except Exception as e:
                corrected_text = raw_text + "\n\n[Note: Mistral correction failed. Raw output shown.]"

            # Generate PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in corrected_text.split('\n'):
                pdf.multi_cell(0, 10, line)

            pdf_path = "corrected_transcript.pdf"
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Click to Download PDF", f, file_name="transcription.pdf")

    st.markdown('</div>', unsafe_allow_html=True)
