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

# Initialize session state
if 'raw_transcript' not in st.session_state:
    st.session_state.raw_transcript = ""
if 'corrected_transcript' not in st.session_state:
    st.session_state.corrected_transcript = ""
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Recorder UI
audio = mic_recorder(
    start_prompt="🔴 Start Recording", 
    stop_prompt="⏹ Stop Recording", 
    key="recorder",
    format="wav"
)

# Process audio when recording stops
if audio:
    st.info("⏳ Transcribing audio...")
    
    # Reset results when new recording starts
    st.session_state.show_results = False
    st.session_state.corrected_transcript = ""

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio["bytes"])
        audio_path = tmpfile.name

    # Transcribe using Whisper
    try:
        segments, _ = model.transcribe(audio_path)
        st.session_state.raw_transcript = " ".join([s.text for s in segments])
        
        # Clean up temp file
        os.unlink(audio_path)
        
        # Display raw transcript immediately
        st.subheader("📝 Raw Transcription")
        st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, key="raw_display")
        
        # Process with Mistral if we have text
        if st.session_state.raw_transcript.strip():
            with st.spinner("🤖 Processing with Mistral AI..."):
                # Correct using Mistral
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system", "content": "Correct transcription errors, improve grammar, and fix punctuation. Maintain the original meaning and structure."},
                        {"role": "user", "content": st.session_state.raw_transcript}
                    ],
                    "temperature": 0.3
                }

                try:
                    res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
                    res.raise_for_status()
                    st.session_state.corrected_transcript = res.json()["choices"][0]["message"]["content"]
                    st.session_state.show_results = True
                except Exception as e:
                    st.error(f"❌ Mistral correction failed: {str(e)}")
                    st.session_state.corrected_transcript = st.session_state.raw_transcript
                    st.session_state.show_results = True
        else:
            st.warning("No speech detected in the recording.")
            
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        if 'audio_path' in locals():
            os.unlink(audio_path)

# Display results after processing
if st.session_state.show_results and st.session_state.corrected_transcript:
    st.subheader("✅ Corrected Transcript")
    st.text_area("Corrected Transcript", st.session_state.corrected_transcript, height=200, key="corrected_display")
    
    # Calculate and display Jaccard Similarity
    if st.session_state.raw_transcript and st.session_state.corrected_transcript:
        similarity_score = jaccard_similarity(st.session_state.raw_transcript, st.session_state.corrected_transcript)
        
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
        
        # Create columns for better layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="Jaccard Similarity", 
                value=f"{similarity_score:.3f}",
                help="Measures word overlap between original and corrected transcripts (0.0 = no overlap, 1.0 = identical)"
            )
        
        with col2:
            if similarity_score >= 0.8:
                st.success(f"**{interpretation}**")
            elif similarity_score >= 0.6:
                st.warning(f"**{interpretation}**")
            else:
                st.error(f"**{interpretation}**")
        
        # Additional statistics
        with st.expander("📈 Detailed Analysis"):
            original_words = len(st.session_state.raw_transcript.split())
            corrected_words = len(st.session_state.corrected_transcript.split())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Words", original_words)
            with col2:
                st.metric("Corrected Words", corrected_words)
            with col3:
                word_change = corrected_words - original_words
                st.metric("Word Count Change", word_change, delta=word_change)
            
            # Show word overlap details
            st.subheader("Word Analysis")
            raw_words = set(st.session_state.raw_transcript.lower().split())
            corrected_words_set = set(st.session_state.corrected_transcript.lower().split())
            
            intersection = raw_words.intersection(corrected_words_set)
            only_in_raw = raw_words - corrected_words_set
            only_in_corrected = corrected_words_set - raw_words
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Common Words", len(intersection))
            with col2:
                st.metric("Only in Original", len(only_in_raw))
            with col3:
                st.metric("Only in Corrected", len(only_in_corrected))

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
        for line in st.session_state.raw_transcript.split("\n"):
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

# Show current state for debugging
if st.checkbox("Show Debug Info"):
    st.write("**Session State:**")
    st.write(f"Raw transcript length: {len(st.session_state.raw_transcript)}")
    st.write(f"Corrected transcript length: {len(st.session_state.corrected_transcript)}")
    st.write(f"Show results: {st.session_state.show_results}")
    st.write(f"Audio data available: {audio is not None}")
