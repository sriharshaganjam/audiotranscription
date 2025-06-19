import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import tempfile
from fpdf import FPDF
import os
import requests
from dotenv import load_dotenv
import numpy as np
from faster_whisper import WhisperModel

# Load API key from environment
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="Live Audio Transcriber", layout="wide")
st.title("🎙️ Live Audio Transcriber with Mistral Correction")

# Load Whisper model (tiny is fast)
model = WhisperModel("tiny", compute_type="int8")

# Streamlit session state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "corrected" not in st.session_state:
    st.session_state.corrected = ""
if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False

# Audio Processor
class AudioProcessor:
    def __init__(self) -> None:
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.int16).tobytes()
        self.frames.append(audio)
        return frame

# Start mic input stream
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    audio_receiver_size=4096,
    audio_processor_factory=AudioProcessor,
)

# Button to trigger transcription
if st.button("🎬 Transcribe"):
    st.info("⏳ Transcribing...")

    if webrtc_ctx.state.playing:
        audio_processor = webrtc_ctx.audio_processor
        if audio_processor and hasattr(audio_processor, "frames"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                audio_path = f.name
                f.write(b"".join(audio_processor.frames))

            # Run Whisper transcription
            segments, _ = model.transcribe(audio_path)
            full_text = " ".join([s.text for s in segments])
            st.session_state.transcript = full_text
            st.text_area("📝 Raw Transcript", full_text, height=200)

            # Call Mistral API to correct text
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
                st.session_state.show_pdf = True
                st.text_area("✅ Corrected Transcript", corrected, height=200)
            except Exception as e:
                st.error("❌ Mistral correction failed. Showing raw text.")
                st.session_state.corrected = full_text
                st.session_state.show_pdf = True
        else:
            st.warning("🎙️ No audio captured. Please speak and try again.")
    else:
        st.warning("🔴 Start the microphone before transcribing.")

# PDF download
if st.session_state.show_pdf:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in st.session_state.corrected.split("\n"):
        pdf.multi_cell(0, 10, line)

    with open("transcript.pdf", "wb") as f:
        pdf.output(f)

    with open("transcript.pdf", "rb") as f:
        st.download_button("📄 Download Transcript as PDF", f, file_name="transcript.pdf")
