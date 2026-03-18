import gradio as gr
import librosa
import numpy as np
import joblib

model = joblib.load("random_forest_model.pkl")

def extract_features(file_path):
    try:
        nfft = 512
        target_length = 200

        y, sr = librosa.load(file_path, sr=None)
        if len(y) < 1024:
            y = np.pad(y, (0, 1024 - len(y)), mode='constant')
        chroma = librosa.feature.chroma_stft(y=y, sr=sr,n_fft=nfft)
        mean_value = np.mean(chroma)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma = np.pad(chroma, ((0, 0), (0, max(0, target_length - chroma.shape[1]))), mode='constant',constant_values=mean_value)[:, :target_length]

        mfccs = librosa.feature.mfcc(y=y, sr=sr,n_fft=nfft)
        mean_value = np.mean(mfccs)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, target_length - mfccs.shape[1]))), mode='constant',constant_values=mean_value)[:, :target_length]

        spect_band = librosa.feature.spectral_bandwidth(y=y,sr=sr,n_fft=nfft)
        mean_value = np.mean(spect_band)
        specb_mean = np.mean(spect_band.T, axis=0)
        spect_band = np.pad(spect_band, ((0, 0), (0, max(0, target_length - spect_band.shape[1]))), mode='constant',constant_values=mean_value)[:,:target_length]

        spect_cen = librosa.feature.spectral_centroid(y=y,sr=sr,n_fft=nfft)
        mean_value = np.mean(spect_cen)
        specc_mean = np.mean(spect_cen.T,axis=0)
        spect_cen = np.pad(spect_cen, ((0, 0), (0, max(0, target_length - spect_cen.shape[1]))), mode='constant',constant_values=mean_value)[:,:target_length]

        spect_flat = librosa.feature.spectral_flatness(y=y,n_fft=nfft)
        mean_value = np.mean(spect_flat)
        specf_mean = np.mean(spect_flat.T , axis=0)
        spect_flat = np.pad(spect_flat, ((0, 0), (0, max(0, target_length - spect_flat.shape[1]))), mode='constant',constant_values=mean_value)[:,:target_length]

        spect_con = librosa.feature.spectral_contrast(y=y,sr=sr,n_fft=nfft)
        mean_value = np.mean(spect_con)
        specco_mean = np.mean(spect_con.T, axis=0)
        spect_con = np.pad(spect_con, ((0, 0), (0, max(0, target_length - spect_con.shape[1]))), mode='constant',constant_values=mean_value)[:, :target_length]

        tonz = librosa.feature.tonnetz(y=y,sr=sr)
        mean_value = np.mean(tonz)
        ton_mean = np.mean(tonz.T, axis = 0)
        tonz = np.pad(tonz, ((0, 0), (0, max(0, target_length - tonz.shape[1]))), mode='constant',constant_values=mean_value)[:, :target_length]

        times = librosa.times_like(spect_cen)

        return {
            'y': y,
            'sr': sr,
            'chroma_mean': chroma_mean,
            'mfccs_mean': mfccs_mean,
            'specb_mean': specb_mean,
            'specc_mean': specc_mean,
            'specf_mean': specf_mean,
            'specco_mean': specco_mean,
            'ton_mean': ton_mean,
            'times': times
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict(audio_file):
    features = extract_features(audio_file)
 
    if features is None:
        return "⚠️ Error processing audio", {"Real": 0.0, "Fake": 0.0}
 
    meanlist = np.hstack([
        features['chroma_mean'],
        features['mfccs_mean'],
        features['specb_mean'],
        features['specc_mean'],
        features['specf_mean'],
        features['specco_mean'],
        features['ton_mean']
    ]).reshape(1, -1)  # reshape to (1, n_features) for sklearn
 
    proba = model.predict_proba(meanlist)[0]  # fix: was passing features dict before
    fake_conf = float(proba[0])
    real_conf = float(proba[1])
    label = "🔴 Fake Audio" if fake_conf > 0.5 else "🟢 Real Audio"
    return label, {"Real": real_conf, "Fake": fake_conf}
 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="Fake Audio Detector",
    description="Upload an audio file to detect whether it's real or AI-generated/fake.",
    examples=[]
)
 
demo.launch()
 
