import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Speech Emotion Dashboard",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Speech Emotion Recognition Dashboard")
st.caption("Signal Analysis • Feature Visualization • Deep Learning Inference")

# =====================================================
# Load model (cached)
# =====================================================
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "models/wav2vec2_cremad"
    )
    model.eval()
    return processor, model

processor, model = load_model()

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# =====================================================
# Audio + plotting utilities
# =====================================================
def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def plot_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color="cyan")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

def plot_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(mfcc, cmap="magma", ax=ax)
    ax.set_title("MFCC Heatmap")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("MFCC Index")
    return fig

def plot_spectrogram(audio, sr):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        cmap="inferno",
        ax=ax
    )
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

# =====================================================
# Upload section
# =====================================================
st.markdown("### 📤 Upload Speech Audio")
audio_file = st.file_uploader(
    "Upload a WAV file (16 kHz recommended)",
    type=["wav"]
)

if audio_file:
    st.markdown("---")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    audio = load_audio(tmp_path, 16000)

    # =================================================
    # Prediction
    # =================================================
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

    # =================================================
    # Top summary row
    # =================================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Emotion", labels[pred_id].upper())

    with col2:
        st.metric("Confidence", f"{probs[pred_id].item():.2f}")

    with col3:
        st.audio(audio_file)

    st.markdown("---")

    # =================================================
    # Signal & Feature dashboard
    # =================================================
    col_left, col_right = st.columns(2)

    with col_left:
        st.pyplot(plot_waveform(audio, 16000))

    with col_right:
        st.pyplot(plot_mfcc(audio, 16000))

    st.pyplot(plot_spectrogram(audio, 16000))

    # =================================================
    # Probability distribution
    # =================================================
    st.markdown("### 📊 Emotion Probability Distribution")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(labels, probs.numpy(), color="#FF4B4B")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Softmax Output")
    st.pyplot(fig)

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption(
    "Dashboard-style Speech Emotion Recognition | Wav2Vec2 + Streamlit"
)
