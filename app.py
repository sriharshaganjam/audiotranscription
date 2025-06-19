import streamlit as st
import tempfile
import io
from fpdf import FPDF
import requests
import base64
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel

# App title
st.set_page_config(page_title="Live Audio Transcription", layout="wide")

# Session state variables
if "recording" not in st.session_state:
    st.session_state.recording = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "corrected" not in st.session_state:
    st.session_state.corrected = ""

# Main layout
st.title("🎙️ Live Audio Transcription App")

# Transcript display area with auto-scroll
st.markdown("### 📝 Live Transcript")
transcript_box = st.empty()
transcript_box.markdown(st.session_state.corrected or st.session_state.transcript or "_Waiting for transcription..._")

# Bottom fixed container for buttons and controls
with st.container():
    st.markdown("""
        <style>
        .fixed-bottom {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f9f9f9;
            padding: 1rem;
            border-top: 1px solid #ccc;
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)

        # Button logic
        if not st.session_state.recording:
            if st.button("🔴 Transcribe", key="start"):
                st.session_state.recording = True
                st.rerun()
        else:
            if st.button("🛑 Stop", key="stop"):
                st.session_state.recording = False
                st.rerun()

        # Only show when stopped
        if not st.session_state.recording and st.session_state.corrected:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in st.session_state.corrected.split("\n"):
                pdf.multi_cell(0, 10, line)

            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)

            st.download_button(
                "📄 Download Transcript as PDF",
                data=pdf_buffer,
                file_name="transcript.pdf",
                mime="application/pdf"
            )

        st.markdown('</div>', unsafe_allow_html=True)

# Audio recorder logic
if st.session_state.recording:
    audio = mic_recorder(
        start_prompt="Recording... Speak now!",
        stop_prompt="Click stop above to finish.",
        key="mic",
        just_once=False,
        use_container_width=True,
    )

    if audio:
        st.session_state.audio_data = audio["bytes"]

# Transcribe and correct once stopped
if not st.session_state.recording and st.session_state.audio_data and not st.session_state.corrected:
    with st.spinner("Transcribing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(st.session_state.audio_data)
            tmp_path = tmp.name

        # Load model
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path)
        transcript = " ".join([seg.text for seg in segments])
        st.session_state.transcript = transcript

    with st.spinner("Correcting with Mistral..."):
        prompt = f"Correct this transcribed text without changing its meaning:\n\n{transcript}"
        headers = {
            "Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-small",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
        st.session_state.corrected = response.json()["choices"][0]["message"]["content"]

        transcript_box.markdown(st.session_state.corrected)
