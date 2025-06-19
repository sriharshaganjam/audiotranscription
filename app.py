import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import tempfile
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Load API key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="Live Audio Transcriber", layout="wide")
st.title("🎙️ Live Audio Transcriber with Mistral Correction")

# Whisper Model
model = WhisperModel("tiny", compute_type="int8")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "corrected" not in st.session_state:
    st.session_state.corrected = ""
if "frames" not in st.session_state:
    st.session_state.frames = []
if "recording" not in st.session_state:
    st.session_state.recording = False

# Audio processor for collecting audio frames
class AudioProcessor:
    def __init__(self):
        self.recording = True

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.int16).tobytes()
        if st.session_state.recording:
            st.session_state.frames.append(audio)
        return frame

# Start WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="sendonly-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Sticky button bar
st.markdown(
    """
    <style>
    .fixed-button {
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        background-color: red;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        z-index: 9999;
        font-weight: bold;
        cursor: pointer;
    }
    .fixed-pdf {
        position: fixed;
        bottom: 1rem;
        left: 200px;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])
with col1:
    if not st.session_state.recording:
        if st.button("🔴 Transcribe", key="start_button"):
            st.session_state.recording = True
            st.session_state.frames = []
    else:
        if st.button("⏹ Stop", key="stop_button"):
            st.session_state.recording = False
            st.info("⏳ Transcribing...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                audio_path = f.name
                f.write(b"".join(st.session_state.frames))

            # Whisper transcription
            segments, _ = model.transcribe(audio_path)
            full_text = " ".join([s.text for s in segments])
            st.session_state.transcript = full_text

            # Mistral correction
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "mistral-tiny",
                "messages": [
                    {"role": "system", "content": "Correct transcription errors and grammar."},
                    {"role": "user", "content": full_text}
                ],
                "temperature": 0.3
            }

            try:
                r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
                r.raise_for_status()
                corrected = r.json()["choices"][0]["message"]["content"]
                st.session_state.corrected = corrected
            except Exception as e:
                st.error("❌ Mistral correction failed. Showing raw text.")
                st.session_state.corrected = full_text

            st.text_area("✅ Final Transcript", st.session_state.corrected, height=300)

# Download button
if st.session_state.corrected:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in st.session_state.corrected.split("\n"):
        pdf.multi_cell(0, 10, line)

    with open("transcript.pdf", "wb") as f:
        pdf.output(f)

    with open("transcript.pdf", "rb") as f:
        st.download_button("📄 Download Transcript as PDF", f, file_name="transcript.pdf")
