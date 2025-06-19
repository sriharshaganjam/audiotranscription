import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import tempfile
from fpdf import FPDF
import os
import requests
from dotenv import load_dotenv
import numpy as np
from faster_whisper import WhisperModel

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="Live Transcriber", layout="wide")
st.title("🎤 Live Audio Transcriber with Mistral Correction")

# Load model (tiny for fast use)
model = WhisperModel("tiny", compute_type="int8")

# Streamlit session state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "corrected" not in st.session_state:
    st.session_state.corrected = ""
if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False

# Audio processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        audio = audio.flatten().astype(np.int16).tobytes()
        self.frames.append(audio)
        return frame

processor = AudioProcessor()

# WebRTC mic input
webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    client_settings=ClientSettings(
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    audio_receiver_size=4096,
    audio_processor_factory=lambda: processor,
)

if st.button("Transcribe"):
    st.info("🧠 Transcribing...")

    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_path = f.name
        f.write(b"".join(processor.frames))

    segments, info = model.transcribe(audio_path)
    text = " ".join([segment.text for segment in segments])
    st.session_state.transcript = text
    st.text_area("Raw Transcript", text, height=200)

    # Call Mistral API for correction
    st.info("✨ Sending to Mistral for correction...")
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that fixes grammar and transcription errors."},
            {"role": "user", "content": st.session_state.transcript}
        ],
        "temperature": 0.3
    }

    try:
        r = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)
        r.raise_for_status()
        corrected = r.json()["choices"][0]["message"]["content"]
        st.session_state.corrected = corrected
        st.text_area("✅ Corrected Transcript", corrected, height=200)
        st.session_state.show_pdf = True
    except Exception as e:
        st.error("⚠️ Mistral correction failed.")
        st.session_state.corrected = st.session_state.transcript
        st.session_state.show_pdf = True

# Generate and download PDF
if st.session_state.show_pdf:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in st.session_state.corrected.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf_path = "transcript.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Transcript as PDF", f, file_name="transcript.pdf")
