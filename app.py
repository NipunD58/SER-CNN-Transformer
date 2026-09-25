"""
Gradio Speech Emotion Recognition App
Deploy on Hugging Face Spaces or run locally.

Usage (local):
    python app.py
"""

import os
import warnings
import numpy as np
import torch
import gradio as gr
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predict import extract_mfcc_features, load_trained_model, EMOTION_CLASSES

# Suppress librosa audioread deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed")

# ── Configuration ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "IEMOCAP", "model_CTMAM_mfcc_all.pth",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_EMOJI = {"Neutral": "😐", "Happy": "😄", "Sad": "😢", "Angry": "😠"}

# ── Load model once at startup ───────────────────────────────────────
print(f"[SER] Loading model from {MODEL_PATH} on {DEVICE} …")
model = load_trained_model(model_path=MODEL_PATH, shape=(26, 57), device=DEVICE)
print("[SER] Model loaded ✓")


# ── Visualisation helpers ────────────────────────────────────────────
def plot_waveform(wav, sr):
    fig, ax = plt.subplots(figsize=(6, 3))
    times = np.linspace(0, len(wav) / sr, num=len(wav))
    ax.plot(times, wav, linewidth=0.5)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_spectrogram(wav, sr):
    fig, ax = plt.subplots(figsize=(6, 3))
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def load_audio(filepath):
    """Load audio from any format, always returns (wav, 16000)."""
    try:
        wav, sr = sf.read(filepath)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return wav, 16000
    except Exception:
        wav, sr = librosa.load(filepath, sr=16000)
        return wav, 16000


def make_playable_wav(wav, sr):
    """Save wav array to a temp .wav file for browser playback."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, sr)
    return tmp.name


# ── Core prediction function ─────────────────────────────────────────
def predict_emotion_gradio(audio):
    if audio is None:
        return None, None, None, "⚠️ No audio provided.", None

    # Load and resample audio
    wav, sr = load_audio(audio)

    if len(wav) == 0:
        return None, None, None, "⚠️ Audio is empty.", None

    # Create a playable wav for the browser
    playable_path = make_playable_wav(wav, sr)

    # Plots
    fig_wave = plot_waveform(wav, sr)
    fig_spec = plot_spectrogram(wav, sr)

    # Extract MFCC features
    features = extract_mfcc_features(wav, sample_rate=16000, nmfcc=26)

    # Inference
    x = torch.from_numpy(features).float().to(DEVICE)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = torch.cat((x, x), 0)

    with torch.no_grad():
        out = model(x.unsqueeze(1))
        probs = torch.nn.functional.softmax(out, dim=-1).mean(dim=0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = EMOTION_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Label dict for gr.Label
    label_output = {e: float(p) for e, p in zip(EMOTION_CLASSES, probs)}

    # Result text
    emoji = EMOTION_EMOJI.get(pred_label, "")
    result_text = f"{emoji} **{pred_label}** — {confidence * 100:.1f}% confidence"

    return playable_path, fig_wave, fig_spec, result_text, label_output


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="Speech Emotion Recognition") as demo:

    gr.Markdown("# 🎙️ Speech Emotion Recognition")
    gr.Markdown("CNN-Transformer + Multidimensional Attention · IEMOCAP · 4 emotions")

    with gr.Row():
        audio_input = gr.Audio(
            label="Upload Audio",
            type="filepath",
            sources=["upload", "microphone"],
        )
        wave_plot = gr.Plot(label="Waveform")
        spec_plot = gr.Plot(label="Spectrogram")

    predict_btn = gr.Button("Analyse Emotion", variant="primary")

    with gr.Row():
        audio_playback = gr.Audio(label="Playback", type="filepath", interactive=False)

    with gr.Row():
        result_text = gr.Markdown(label="Prediction")
        label_output = gr.Label(label="Emotion Probabilities", num_top_classes=4)

    predict_btn.click(
        fn=predict_emotion_gradio,
        inputs=[audio_input],
        outputs=[audio_playback, wave_plot, spec_plot, result_text, label_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
