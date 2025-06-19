import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from io import BytesIO
from fpdf import FPDF
import requests
import os

# Set page config
st.set_page_config(page_title="Live Audio Transcriber", layout="wide")

# Inject CSS to fix the Transcribe/Stop button at the bottom
st.markdown("""
    <style>
        #fixed-buttons {
            position: fixed;
            bottom: 20px;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 10px 0;
            border-top: 1px solid #ddd;
            text-align: center;
            z-index: 9999;
        }
        .element-container:has(#fixed-buttons) {
            padding-bottom: 100px;
        }
    </style>
""", unsafe_allow_html=True)

# Init session state
for key in ["recording", "audio_data", "transcript", "corrected"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "audio_data" else False if key == "recording" else ""

# Load Whisper model once
@st.cache_resource
def load_model():
    return WhisperModel("tiny", compute_type="int8")

model = load_model()

# Heading
st.title("🎙️ Live Audio Transcriber with Mistral Correction")

# Display transcription result
transcript_box = st.empty()

# Mic Recorder
audio = mic_recorder(key="recorder")

if audio:
    st.session_state.audio_data = audio["bytes"]
    st.success("✅ Audio recorded! Hit Stop to transcribe.")

# Handle transcription on Stop
if not st.session_state.recording and st.session_state.audio_data and not st.session_state.transcript:
    with st.spinner("Transcribing..."):
        with BytesIO(st.session_state.audio_data) as audio_file:
            segments, _ = model.transcribe(audio_file, beam_size=5)
            result = " ".join([seg.text.strip() for seg in segments])
            st.session_state.transcript = result

        # Call Mistral for correction
        mistral_prompt = f"Correct the grammar and punctuation of the following spoken text:\n\n{st.session_state.transcript}"
        headers = {
            "Authorization": f"Bearer {os.environ.get('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json={
                "model": "mistral-tiny",
                "messages": [{"role": "user", "content": mistral_prompt}]
            }
        )
        if response.status_code == 200:
            st.session_state.corrected = response.json()["choices"][0]["message"]["content"].strip()
        else:
            st.warning("Failed to correct transcript using Mistral.")

# Update transcript display
if st.session_state.corrected:
    transcript_box.markdown(st.session_state.corrected)
elif st.session_state.transcript:
    transcript_box.markdown(st.session_state.transcript)
elif st.session_state.recording:
    transcript_box.markdown("_🎤 Listening... Speak into the mic._")
elif st.session_state.audio_data and not st.session_state.transcript:
    transcript_box.markdown("_⚠️ Transcription failed or returned no text._")
else:
    transcript_box.markdown("_Waiting for transcription..._")

# Fixed control buttons
with st.container():
    st.markdown('<div id="fixed-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if not st.session_state.recording:
            if st.button("🔴 Transcribe", key="start"):
                st.session_state.recording = True
                st.session_state.audio_data = None
                st.session_state.transcript = ""
                st.session_state.corrected = ""
        else:
            if st.button("🛑 Stop", key="stop"):
                st.session_state.recording = False

    with col2:
        if st.session_state.corrected:
            # Create PDF in memory
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in st.session_state.corrected.split("\n"):
                pdf.multi_cell(0, 10, line)

            buffer = BytesIO()
            pdf.output(buffer)
            buffer.seek(0)

            st.download_button("📄 Download Transcript as PDF", buffer, file_name="transcript.pdf")

    st.markdown('</div>', unsafe_allow_html=True)
