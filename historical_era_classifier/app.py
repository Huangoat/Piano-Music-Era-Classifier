import streamlit as st
import joblib
import pandas as pd
import tempfile
import os
from pathlib import Path
import torch

from pydub import AudioSegment
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
from audio_recorder_streamlit import audio_recorder

from src.realtime_feature_extracter import extract_features_from_midi

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = (PROJECT_ROOT / "models" / "model.pkl")

# Page config
st.set_page_config(
    page_title="Classical Piano Music Era Classifier",
    page_icon="🎼",
    layout="centered"
)

st.title("🎼 Classical Piano Music Era Classifier")
@st.cache_resource
def load_classifier_model():
    return joblib.load(MODEL_DIR)

@st.cache_resource
def load_transcription_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = PROJECT_ROOT / "src" / "piano_transcription_inference_data" / \
                      "note_F1=0.9677_pedal_F1=0.9186.pth"
    return PianoTranscription(device=device, checkpoint_path=checkpoint_path)


# Initialize models
try:
    classifier_model = load_classifier_model()

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Input Settings")
input_type = st.sidebar.radio(
    "Choose Input Method:",
    ["Upload MIDI", "Upload MP3/Audio", "Record Audio"]
)

st.write(f"Currently using: {input_type}")
st.write("Note: This model is optimized for **solo piano**. Orchestral music or music with more than 1 instrument playing will likely yield poor results.")

#Turning audio into midi 
def transcode_audio_to_midi(audio_path):
    try:
        transcriptor = load_transcription_model()
        audio_snippet = AudioSegment.from_file(audio_path)
        cut = 40 * 1000 
    
        if len(audio_snippet) > cut:
            st.warning("Cropping to the first 40 seconds for faster era analysis.")
            audio_snippet = audio_snippet[:cut]
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_snippet:
            audio_snippet.export(tmp_snippet.name, format="wav")
            processed_audio_path = tmp_snippet.name

        (audio, _) = load_audio(processed_audio_path, sr=sample_rate, mono=True)
        
        temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi_out_path = temp_midi.name
        temp_midi.close() 
        transcriptor.transcribe(audio, midi_out_path)

        os.remove(processed_audio_path)
        
        return midi_out_path
        
    except Exception as e:
        st.error(f"Transcription Failed: {e}")
        return None

#Uses model to classift
def run_prediction(midi_path):
    try:
        #Feature Extraction
        features = extract_features_from_midi(midi_path)
        
        if features is None or features.empty:
            st.error("Could not extract features from this MIDI file.")
            return
        # Using model to predict
        prediction = classifier_model.predict(features)[0]
        
        st.subheader("🎶 Predicted Era")
        st.success(f"**{prediction}**")

        #Probability breakdown
        if hasattr(classifier_model, "predict_proba"):
            probs = classifier_model.predict_proba(features)[0]
            classes = classifier_model.classes_
            
            confidence = probs.max()
            st.write(f"**Confidence:** {confidence:.2%}")

            prob_df = pd.DataFrame({
                "Era": classes,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Era"))
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

midi_to_process = None

# Midi upload
if input_type == "Upload MIDI":
    uploaded_file = st.file_uploader("Upload a MIDI file", type=["mid", "midi"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
            tmp.write(uploaded_file.read())
            midi_to_process = tmp.name

# Audio upload
elif input_type == "Upload MP3/Audio":
    uploaded_audio = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name
            
            with st.spinner("Transcribing audio to MIDI (this may take a moment)..."):
                midi_to_process = transcode_audio_to_midi(tmp_audio_path)
            
            os.remove(tmp_audio_path)

# Recording upload
elif input_type == "Record Audio":
    st.write("Click the mic to record your piano playing:")
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100)
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name
            
            with st.spinner("Transcribing recording..."):
                midi_to_process = transcode_audio_to_midi(tmp_audio_path)

            os.remove(tmp_audio_path)

# Final execution
if midi_to_process:
    if input_type != "Upload MIDI":
        with open(midi_to_process, "rb") as f:
            st.download_button("Download Generated MIDI", f, file_name="converted_piano.mid")

    run_prediction(midi_to_process)
    
    if os.path.exists(midi_to_process):
        os.remove(midi_to_process)
