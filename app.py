import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper
import av
import tempfile
from fpdf import FPDF
import os
import requests
from dotenv import load_dotenv
import numpy as np

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="Web Audio Transcriber", layout="wide")
st.title("🎤 Web-Based Audio Transcriber with Mistral Cleanup")

# Load Whisper model
model = whisper.load_model("base")

# Session state
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
    st.info("Processing audio...")

    # Save recorded bytes to temp wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wf = f.name
        with open(wf, "wb") as out:
            out.write(b"".join(processor.frames))

    # Whisper transcription
    result = model.transcribe(wf)
    st.session_state.transcript = result["text"]
    st.text_area("Transcribed Text", st.session_state.transcript, height=200)

    # Mistral Correction
    st.info("Sending to Mistral for correction...")
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "Correct grammar and transcription errors."},
            {"role": "user", "content": st.session_state.transcript}
        ],
        "temperature": 0.3
    }

    try:
        r = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)
        r.raise_for_status()
        corrected = r.json()["choices"][0]["message"]["content"]
        st.session_state.corrected = corrected
        st.session_state.show_pdf = True
        st.text_area("Corrected Transcript", corrected, height=200)
    except Exception as e:
        st.error("Mistral correction failed.")
        st.session_state.corrected = st.session_state.transcript
        st.session_state.show_pdf = True

# PDF Export
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
