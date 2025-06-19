import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import tempfile
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Whisper model
model = WhisperModel("tiny", compute_type="int8")

st.set_page_config(page_title="Live Audio Transcriber", layout="centered")
st.title("🎤 Live Audio Transcriber with Mistral Correction")

# Start/Stop Recorder
audio = mic_recorder(start_prompt="🔴 Transcribe", stop_prompt="⏹ Stop", key="recorder")

# If audio recorded
if audio:
    st.info("⏳ Transcribing...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio["bytes"])
        audio_path = tmpfile.name

    # Whisper Transcription
    segments, _ = model.transcribe(audio_path)
    full_text = " ".join([s.text for s in segments])
    st.subheader("📝 Raw Transcription")
    st.text_area("Transcript", full_text, height=200)

    # Mistral Correction
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "Correct transcription errors and grammar."},
            {"role": "user", "content": full_text}
        ],
        "temperature": 0.3
    }

    try:
        r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        corrected = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error("❌ Mistral correction failed. Showing raw transcript.")
        corrected = full_text

    st.subheader("✅ Final Corrected Transcript")
    st.text_area("Corrected Transcript", corrected, height=200)

    # PDF Download
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in corrected.split("\n"):
        pdf.multi_cell(0, 10, line)

    with open("transcript.pdf", "wb") as f:
        pdf.output(f)

    with open("transcript.pdf", "rb") as f:
        st.download_button("📄 Download Transcript as PDF", f, file_name="transcript.pdf")
