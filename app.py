import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import tempfile
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv
import threading
import time

# Load Mistral API Key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Load Whisper model
model = WhisperModel("tiny", compute_type="int8")

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate intersection and union
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Handle empty texts
    if len(union) == 0:
        return 1.0 if len(words1) == 0 and len(words2) == 0 else 0.0
    
    return len(intersection) / len(union)

st.set_page_config(page_title="Live Audio Transcriber", layout="centered")
st.title("🎤 Live Audio Transcriber with Mistral Correction")

# Initialize session state for live transcription
if 'live_transcript' not in st.session_state:
    st.session_state.live_transcript = ""
if 'final_transcript' not in st.session_state:
    st.session_state.final_transcript = ""
if 'corrected_transcript' not in st.session_state:
    st.session_state.corrected_transcript = ""
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False

# Recorder UI
audio = mic_recorder(
    start_prompt="🔴 Start Recording", 
    stop_prompt="⏹ Stop Recording", 
    key="recorder",
    format="wav",
    use_container_width=True
)

# Live transcript display (updates while recording)
st.subheader("📝 Live Transcription")
live_placeholder = st.empty()

# Check if recording just started
if audio and audio.get('bytes') and not st.session_state.recording_active:
    st.session_state.recording_active = True
    st.session_state.live_transcript = ""
    st.session_state.corrected_transcript = ""

# Live transcription during recording
if st.session_state.recording_active and audio and audio.get('bytes'):
    try:
        # Save current audio chunk temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio["bytes"])
            audio_path = tmpfile.name

        # Transcribe current chunk
        segments, _ = model.transcribe(audio_path)
        current_text = " ".join([s.text for s in segments])
        
        # Update live transcript
        st.session_state.live_transcript = current_text
        
        # Clean up temp file
        os.unlink(audio_path)
        
    except Exception as e:
        st.error(f"Live transcription error: {str(e)}")

# Display live transcript
with live_placeholder.container():
    st.text_area(
        "Live Transcript (updating as you speak)", 
        st.session_state.live_transcript, 
        height=200,
        key="live_display"
    )

# Process final transcript when recording stops
if not audio or not audio.get('bytes'):
    if st.session_state.recording_active:
        st.session_state.recording_active = False
        st.session_state.final_transcript = st.session_state.live_transcript
        
        # Only process correction if we have text
        if st.session_state.final_transcript.strip():
            st.info("⏳ Processing with Mistral AI...")
            
            # Correct using Mistral
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "mistral-tiny",
                "messages": [
                    {"role": "system", "content": "Correct transcription errors, improve grammar, and fix punctuation. Maintain the original meaning and structure."},
                    {"role": "user", "content": st.session_state.final_transcript}
                ],
                "temperature": 0.3
            }

            try:
                res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                st.session_state.corrected_transcript = res.json()["choices"][0]["message"]["content"]
            except Exception as e:
                st.error(f"❌ Mistral correction failed: {str(e)}. Showing raw transcript.")
                st.session_state.corrected_transcript = st.session_state.final_transcript

# Display final results after recording stops
if st.session_state.final_transcript:
    st.subheader("✅ Final Corrected Transcript")
    st.text_area("Corrected Transcript", st.session_state.corrected_transcript, height=200)
    
    # Calculate and display Jaccard Similarity
    if st.session_state.corrected_transcript:
        similarity_score = jaccard_similarity(st.session_state.final_transcript, st.session_state.corrected_transcript)
        
        # Color-code the similarity score
        if similarity_score >= 0.8:
            color = "green"
            interpretation = "Very High - Mistral correction may not be necessary"
        elif similarity_score >= 0.6:
            color = "orange"
            interpretation = "Moderate - Some benefit from Mistral correction"
        else:
            color = "red"
            interpretation = "Low - Mistral correction is beneficial"
        
        st.subheader("📊 Transcript Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Jaccard Similarity Score", 
                value=f"{similarity_score:.3f}",
                help="Measures word overlap between original and corrected transcripts"
            )
        
        with col2:
            st.markdown(f"**Interpretation:** :{color}[{interpretation}]")
        
        # Additional statistics
        with st.expander("📈 Detailed Analysis"):
            original_words = len(st.session_state.final_transcript.split())
            corrected_words = len(st.session_state.corrected_transcript.split())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Words", original_words)
            with col2:
                st.metric("Corrected Words", corrected_words)
            with col3:
                word_change = corrected_words - original_words
                st.metric("Word Count Change", word_change, delta=word_change)

    # Generate PDF with both versions
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Audio Transcription Report", ln=True, align='C')
        pdf.ln(10)
        
        # Similarity score
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Jaccard Similarity Score: {similarity_score:.3f}", ln=True)
        pdf.cell(0, 10, f"Interpretation: {interpretation}", ln=True)
        pdf.ln(5)
        
        # Original transcript
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Original Transcript:", ln=True)
        pdf.set_font("Arial", size=10)
        for line in st.session_state.final_transcript.split("\n"):
            if line.strip():
                try:
                    pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
                except:
                    pdf.multi_cell(0, 8, line.encode('ascii', 'ignore').decode('ascii'))
        pdf.ln(5)
        
        # Corrected transcript
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Corrected Transcript:", ln=True)
        pdf.set_font("Arial", size=10)
        for line in st.session_state.corrected_transcript.split("\n"):
            if line.strip():
                try:
                    pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
                except:
                    pdf.multi_cell(0, 8, line.encode('ascii', 'ignore').decode('ascii'))

        # Save PDF to bytes
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        # Provide download button
        st.download_button(
            label="📄 Download Transcript Report as PDF",
            data=pdf_output,
            file_name="transcript_report.pdf",
            mime="application/pdf"
        )
