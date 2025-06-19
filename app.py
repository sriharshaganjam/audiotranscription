import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import tempfile
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv

# Load Mistral API Key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Load Whisper model
model = WhisperModel("tiny", compute_type="int8")

st.set_page_config(page_title="Live Audio Transcriber", layout="centered")
st.title("🎤 Live Audio Transcriber with Mistral Correction")

# Recorder UI
audio = mic_recorder(start_prompt="🔴 Transcribe", stop_prompt="⏹ Stop", key="recorder")

if audio:
    st.info("⏳ Transcribing...")

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio["bytes"])
        audio_path = tmpfile.name

    # Transcribe using Whisper
    segments, _ = model.transcribe(audio_path)
    full_text = " ".join([s.text for s in segments])
    st.subheader("📝 Raw Transcription")
    st.text_area("Transcript", full_text, height=200)

    # Correct using Mistral
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
        res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        corrected = res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error("❌ Mistral correction failed. Showing raw transcript.")
        corrected = full_text

    st.subheader("✅ Final Corrected Transcript")
    st.text_area("Corrected Transcript", corrected, height=200)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Handle text encoding and line breaks properly
    for line in corrected.split("\n"):
        if line.strip():  # Only add non-empty lines
            # Encode text to handle special characters
            try:
                pdf.multi_cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'))
            except:
                # Fallback for problematic characters
                pdf.multi_cell(0, 10, line.encode('ascii', 'ignore').decode('ascii'))

    # Save PDF to bytes
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    # Provide download button
    st.download_button(
        label="📄 Download Transcript as PDF",
        data=pdf_output,
        file_name="transcript.pdf",
        mime="application/pdf"
    )
