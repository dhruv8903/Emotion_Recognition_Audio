# 🎙️ Speech Emotion Recognition Dashboard

A real-time speech emotion recognition application powered by Wav2Vec2 and Streamlit. Upload audio files and get instant emotion predictions with detailed acoustic visualizations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)

## 🌟 Features

- **Real-time Emotion Prediction**: Upload WAV files and get instant emotion classification
- **6 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad
- **Rich Visualizations**:
  - Waveform display
  - MFCC (Mel-Frequency Cepstral Coefficients) heatmap
  - Log-Mel spectrogram
  - Probability distribution chart
- **Deep Learning Model**: Fine-tuned Wav2Vec2 on CREMA-D dataset
- **Interactive Dashboard**: Clean, intuitive Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd sip_emotion
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
sip_emotion/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── models/
│   └── wav2vec2_cremad/       # Fine-tuned Wav2Vec2 model
└── README.md                   # This file
```

## 🎯 Usage

1. Launch the application using `streamlit run app.py`
2. Click the "Browse files" button to upload a WAV audio file
3. The dashboard will display:
   - Predicted emotion with confidence score
   - Audio player
   - Signal visualizations (waveform, MFCC, spectrogram)
   - Probability distribution across all emotion classes

### Supported Audio Format

- **Format**: WAV
- **Sample Rate**: 16 kHz (recommended)
- **Channels**: Mono or Stereo (automatically converted)

## 🧠 Model Details

- **Base Model**: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- **Fine-tuned On**: CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Architecture**: Wav2Vec2 with sequence classification head
- **Emotions**: 6 classes (angry, disgust, fear, happy, neutral, sad)

## 📊 Technical Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Audio Processing**: Librosa, Soundfile
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Numerical Computing**: NumPy

## 📋 Requirements

```
streamlit
torch
torchaudio
transformers
librosa
soundfile
numpy
safetensors
matplotlib
seaborn
```

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Meta AI for the Wav2Vec2 model
- CREMA-D dataset creators
- Streamlit team for the amazing framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using Streamlit and PyTorch**
