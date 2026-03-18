---
title: Fake Audio Detector
emoji: 🎙️
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🎙️ Fake Audio Detector

A machine learning web app that detects whether an audio clip is **real** or **AI-generated / deepfake**.

Upload any audio file and get an instant prediction with a confidence score.

---

## 🚀 How It Works

1. Upload a `.wav`, `.mp3`, or `.flac` audio file
2. The app extracts a rich set of audio features using `librosa`
3. A trained **Random Forest** classifier predicts whether the audio is real or fake
4. You get a label + confidence score instantly

---

## 🧠 Model & Features

**Classifier:** Random Forest (scikit-learn)

**Features extracted per audio file:**

| Feature | Description |
|---|---|
| MFCC (Mel-Frequency Cepstral Coefficients) | Captures timbral texture of speech |
| Chroma STFT | Represents harmonic/pitch content |
| Spectral Bandwidth | Spread of frequencies around centroid |
| Spectral Centroid | Brightness / center of mass of spectrum |
| Spectral Flatness | Noisiness vs. tonality of signal |
| Spectral Contrast | Peak vs. valley difference across bands |
| Tonnetz | Tonal centroid features (harmonic relations) |

All features are computed as **mean vectors** and concatenated into a single feature vector before classification.

---

## 🛠️ Tech Stack

- **Python** — core language
- **librosa** — audio feature extraction
- **scikit-learn** — Random Forest model
- **Gradio** — web interface
- **Hugging Face Spaces** — deployment

---

## 📁 Project Structure

```
├── app.py                  # Gradio app + prediction logic
├── random_forest_model.pkl # Trained sklearn model
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/fake-audio-detector
cd fake-audio-detector

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

---

## 📦 Requirements

```
gradio
librosa
scikit-learn
numpy
joblib
```

---

## ⚠️ Limitations

- Works best on **speech audio**; music or environmental sounds may yield unreliable results
- Performance depends on the training data distribution — out-of-distribution audio generators may not be detected
- Short clips (< 1 second) are padded but may reduce accuracy

---

