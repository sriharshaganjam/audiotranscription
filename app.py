import streamlit as st
from streamlit_audio_recorder import audio_recorder
import whisper
import tempfile
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv

# Load Mistral API key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Whisper model
model = whisper.load_model("base")

# Streamlit page setup
st.set_page_config(page_title="Live Transcriber", layout="wide")
st.title("🎙️ Audio Transcriber with Mistral Correction")

# State variables
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "corrected" not in st.session_state:
    st.session_state.corrected = ""
if "show_download" not in st.session_state:
    st.session_state.show_download = False

# Audio recording UI
audio_bytes = audio_recorder(text="🔴 Transcribe", recording_color="#FF0000", neutral_color="#6c6c6c", icon_name="microphone", pause_threshold=3.0)

if audio_bytes:
    st.success("✅ Audio recorded. Transcribing...")

    # Save to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    # Transcribe using Whisper
    result = model.transcribe(audio_path, fp16=False)
    st.session_state.transcript = result['text']
    st.text_area("Raw Transcript", st.session_state.transcript, height=200)

    # Mistral cleanup
    st.info("✨ Sending to Mistral for grammar correction...")
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
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        st.session_state.corrected = response.json()['choices'][0]['message']['content']
        st.text_area("✅ Corrected Transcript", st.session_state.corrected, height=200)
        st.session_state.show_download = True
    except Exception as e:
        st.error("❌ Failed to correct transcript. Showing raw transcript only.")
        st.session_state.corrected = st.session_state.transcript
        st.session_state.show_download = True

# Fixed bottom download button
if st.session_state.show_download:
    with st.container():
        st.markdown("""
        <style>
        #fixed-download {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #fff;
            padding: 1rem;
            border-top: 1px solid #ccc;
            z-index: 9999;
            text-align: center;
        }
        </style>
        <div id="fixed-download">
        """, unsafe_allow_html=True)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in st.session_state.corrected.split('\n'):
            pdf.multi_cell(0, 10, line)

        pdf_path = "transcript.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Transcript as PDF", f, file_name="transcript.pdf")

        st.markdown("</div>", unsafe_allow_html=True)
