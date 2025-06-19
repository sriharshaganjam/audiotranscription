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

    # Calculate and display Jaccard Similarity
    similarity_score = jaccard_similarity(full_text, corrected)
    
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

    # Generate PDF
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
    pdf.ln(10)
    
    # Original transcript
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Original Transcript:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    for line in full_text.split("\n"):
        if line.strip():  # Only add non-empty lines
            # Encode text to handle special characters
            try:
                pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
            except:
                # Fallback for problematic characters
                pdf.multi_cell(0, 8, line.encode('ascii', 'ignore').decode('ascii'))
    
    pdf.ln(10)
    
    # Corrected transcript
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Corrected Transcript:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    for line in corrected.split("\n"):
        if line.strip():  # Only add non-empty lines
            # Encode text to handle special characters
            try:
                pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
            except:
                # Fallback for problematic characters
                pdf.multi_cell(0, 8, line.encode('ascii', 'ignore').decode('ascii'))

    # Save PDF to bytes
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    # Provide download button
    st.download_button(
        label="📄 Download Transcript as PDF",
        data=pdf_output,
        file_name="transcript.pdf",
        mime="application/pdf"
    )
