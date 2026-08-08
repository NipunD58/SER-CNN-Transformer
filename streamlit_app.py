"""
Speech Emotion Recognition (SER) - Streamlit Web Application
Backbone: CNN-Transformer with Multidimensional Attention (CTMAM) trained on IEMOCAP
Emotions: Neutral, Happy, Sad, Angry

Usage:
    streamlit run streamlit_app.py
"""

import os
import io
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import tempfile
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try importing predict helpers, else define fallback paths
try:
    from predict import extract_mfcc_features, load_trained_model, EMOTION_CLASSES
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predict import extract_mfcc_features, load_trained_model, EMOTION_CLASSES

# Suppress librosa audioread & soundfile warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed")

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition | CTMAM",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Constants ────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "Neutral": {"desc": "Calm, baseline speaking tone"},
    "Happy":   {"desc": "Positive, cheerful, elevated pitch"},
    "Sad":     {"desc": "Subdued, lower energy, slow tempo"},
    "Angry":   {"desc": "High intensity, elevated volume/pace"},
}

DEFAULT_CHECKPOINT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "IEMOCAP", "model_CTMAM_mfcc_all.pth"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "IEMOCAP", "model_CTMAM_mfcc_all.pth"),
    os.path.join(os.getcwd(), "data", "IEMOCAP", "model_CTMAM_mfcc_all.pth"),
]

def find_default_checkpoint():
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            return os.path.abspath(path)
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


# ── Cached Model Loader ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint_path: str, device_str: str):
    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint not found at: {checkpoint_path}"
    try:
        model = load_trained_model(model_path=checkpoint_path, shape=(26, 57), device=device_str)
        return model, None
    except Exception as e:
        return None, str(e)


# ── Audio Processing Helpers ─────────────────────────────────────────
def load_audio_from_bytes(file_bytes, filename=None):
    """
    Robustly load audio from bytes into a 16kHz mono float32 numpy array.
    Supports .m4a, .aac, .mp3, .wav, .flac, .ogg, .wma, and mic recordings.
    """
    suffix = ""
    if filename and "." in filename:
        suffix = os.path.splitext(filename)[1].lower()
    if not suffix:
        suffix = ".wav"

    # 1. Quick in-memory attempt with soundfile (fast for standard wav/flac/ogg)
    try:
        bio = io.BytesIO(file_bytes)
        wav, sr = sf.read(bio)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return wav, 16000
    except Exception:
        pass

    # 2. Write to a temporary file on disk with proper extension for format-specific decoders
    tmp_path = None
    converted_tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        # 2a. Try torchaudio (natively decodes m4a, mp4, aac, mp3, wav)
        try:
            import torchaudio
            sig, orig_sr = torchaudio.load(tmp_path)
            sig = sig.float()
            if sig.shape[0] > 1:
                sig = torch.mean(sig, dim=0, keepdim=True)
            if orig_sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)
                sig = resampler(sig)
            wav = sig.squeeze().cpu().numpy().astype(np.float32)
            if wav.ndim > 0 and len(wav) > 0:
                return wav, 16000
        except Exception:
            pass

        # 2b. Try pydub (if available)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(tmp_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            raw = audio.get_array_of_samples()
            wav = np.array(raw).astype(np.float32) / float(1 << (8 * audio.sample_width - 1))
            if len(wav) > 0:
                return wav, 16000
        except Exception:
            pass

        # 2c. Try librosa.load on the real filepath (uses audioread / Windows Media Foundation / ffmpeg)
        try:
            wav, sr = librosa.load(tmp_path, sr=16000, mono=True)
            wav = wav.astype(np.float32)
            if len(wav) > 0:
                return wav, 16000
        except Exception:
            pass

        # 2d. Try ffmpeg CLI conversion fallback
        try:
            converted_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            converted_tmp_path = converted_tmp.name
            converted_tmp.close()

            cmd = [
                "ffmpeg", "-y", "-i", tmp_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", converted_tmp_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if res.returncode == 0 and os.path.exists(converted_tmp_path):
                wav, sr = sf.read(converted_tmp_path)
                wav = wav.astype(np.float32)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                return wav, 16000
        except Exception:
            pass

        # 2e. Try audioread directly
        try:
            import audioread
            with audioread.audio_open(tmp_path) as a:
                sr_native = a.samplerate
                nchannels = a.channels
                buf = b"".join(a.read_data())
                wav = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                if nchannels > 1:
                    wav = wav.reshape(-1, nchannels).mean(axis=1)
                if sr_native != 16000:
                    wav = librosa.resample(wav, orig_sr=sr_native, target_sr=16000)
                return wav, 16000
        except Exception:
            pass

        raise RuntimeError(
            "Could not decode audio format. Please ensure ffmpeg or torchaudio is available, or convert the file to .wav or .mp3."
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if converted_tmp_path and os.path.exists(converted_tmp_path):
            try:
                os.remove(converted_tmp_path)
            except Exception:
                pass



def wav_to_bytes(wav, sr=16000):
    """Convert numpy array to WAV bytes."""
    bio = io.BytesIO()
    sf.write(bio, wav, sr, format="WAV")
    return bio.getvalue()


# ── Plotting Helpers ──────────────────────────────────────────────────
def plot_waveform_matplotlib(wav, sr):
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    times = np.linspace(0, len(wav) / sr, num=len(wav))
    ax.plot(times, wav, color="#38bdf8", linewidth=0.8, alpha=0.9)
    ax.set_title("Audio Waveform", color="#f8fafc", fontsize=11, pad=8)
    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig


def plot_spectrogram_matplotlib(wav, sr):
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Mel-Spectrogram", color="#f8fafc", fontsize=11, pad=8)
    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8", labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
    fig.tight_layout()
    return fig


# ── Inference Pipeline ────────────────────────────────────────────────
def run_ser_inference(model, wav, device, segment_length=1.8, overlap=1.6, nmfcc=26):
    """Run model forward pass on extracted MFCC segments."""
    features = extract_mfcc_features(
        wav, sample_rate=16000, nmfcc=nmfcc, segment_length=segment_length, overlap=overlap
    )
    
    x = torch.from_numpy(features).float().to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = torch.cat((x, x), 0)

    with torch.no_grad():
        out = model(x.unsqueeze(1))
        # Per-segment probabilities
        seg_probs = F.softmax(out, dim=-1).cpu().numpy()
        # Mean probabilities across segments
        mean_probs = seg_probs.mean(axis=0)

    pred_idx = int(np.argmax(mean_probs))
    pred_label = EMOTION_CLASSES[pred_idx]
    confidence = float(mean_probs[pred_idx])
    prob_dict = {cls: float(p) for cls, p in zip(EMOTION_CLASSES, mean_probs)}

    return pred_label, confidence, prob_dict, seg_probs, features


# ── Main Application ─────────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown('<div class="main-title">🎙️ Speech Emotion Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deep Learning SER using <b>CTMAM</b> (CNN-Transformer + Multidimensional Attention) trained on the <b>IEMOCAP</b> dataset.</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Checkpoint Selector
        default_ckpt = find_default_checkpoint()
        checkpoint_path = st.text_input(
            "Model Checkpoint (.pth)",
            value=default_ckpt,
            help="Path to trained PyTorch weights file.",
        )
        
        # Device Selector
        cuda_avail = torch.cuda.is_available()
        device_options = ["cuda", "cpu"] if cuda_avail else ["cpu"]
        device_choice = st.selectbox(
            "Inference Device",
            options=device_options,
            index=0,
            help="Select CUDA GPU (if available) or CPU.",
        )

        # Advanced audio slicing parameters
        with st.expander("🛠️ Advanced Extraction Params", expanded=False):
            param_seg_len = st.slider("Segment Window (s)", min_value=1.0, max_value=3.0, value=1.8, step=0.1)
            param_overlap = st.slider("Segment Overlap (s)", min_value=0.5, max_value=1.7, value=1.6, step=0.1)
            param_nmfcc = st.number_input("MFCC Coefficients", min_value=13, max_value=40, value=26)

        st.markdown("---")
        st.markdown("### 🏷️ Target Emotion Classes")
        for emo, cfg in EMOTION_CONFIG.items():
            st.markdown(f"**{cfg['emoji']} {emo}**: *{cfg['desc']}*")



    # ── Load Model ──
    model, load_err = get_cached_model(checkpoint_path, device_choice)
    if load_err or model is None:
        st.error(f"❌ Failed to load model weights: `{load_err}`\nPlease verify checkpoint path in sidebar.")
        return

    # ── Input Method Tabs ──
    tab_upload, tab_record = st.tabs(["📁 Upload File", "🎤 Record Audio"])

    audio_bytes = None
    source_name = "None"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an audio file (.wav, .mp3, .ogg, .flac, .m4a)",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Supported audio formats will be resampled to 16kHz.",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            source_name = uploaded_file.name

    with tab_record:
        st.markdown("Record your voice using your microphone:")
        # Streamlit 1.37+ built-in audio_input
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record speech sample")
            if recorded_audio is not None:
                audio_bytes = recorded_audio.read()
                source_name = "Microphone Recording"
        else:
            st.info("💡 Recording is supported in modern Streamlit or upload a voice recording via 'Upload File'.")

    # ── Main Analysis Section ──
    if audio_bytes is not None:
        try:
            with st.spinner("Processing audio & extracting acoustic features..."):
                wav, sr = load_audio_from_bytes(audio_bytes, filename=source_name)
                duration = len(wav) / sr

                if len(wav) == 0 or duration < 0.1:
                    st.warning("⚠️ The provided audio is empty or too short.")
                    return

                # Convert decoded 16kHz audio to clean WAV bytes for reliable browser playback
                playable_audio_bytes = wav_to_bytes(wav, sr)

                # Run Inference
                pred_label, confidence, prob_dict, seg_probs, mfcc_feat = run_ser_inference(
                    model,
                    wav,
                    device_choice,
                    segment_length=param_seg_len,
                    overlap=param_overlap,
                    nmfcc=param_nmfcc,
                )

            # Audio Player & Quick Metrics
            st.markdown("### 🎧 Audio Playback & Information")
            col_play, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])
            with col_play:
                st.audio(playable_audio_bytes, format="audio/wav")
            with col_m1:
                st.metric("Source", source_name if len(source_name) < 18 else source_name[:15] + "...")
            with col_m2:
                st.metric("Duration", f"{duration:.2f} s")
            with col_m3:
                st.metric("Segments", f"{mfcc_feat.shape[0]} windows")

            st.markdown("---")

            # ── Prediction Results Section ──
            st.markdown("### 🎯 Emotion Recognition Result")
            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                top_cfg = EMOTION_CONFIG.get(pred_label, {"emoji": "🎭", "color": "#38bdf8", "desc": ""})
                st.markdown(
                    f"""
                    <div class="emotion-card" style="border-left: 6px solid {top_cfg['color']};">
                        <div style="font-size: 3.2rem; margin-bottom: 0.2rem;">{top_cfg['emoji']}</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{pred_label}</div>
                        <div class="metric-value" style="color: {top_cfg['color']};">{confidence * 100:.1f}%</div>
                        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">{top_cfg['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("**Emotion Probability Breakdown:**")
                for emo in EMOTION_CLASSES:
                    p = prob_dict.get(emo, 0.0)
                    cfg = EMOTION_CONFIG.get(emo, {"emoji": "", "color": "#38bdf8"})
                    is_winner = emo == pred_label
                    prefix = "👑 " if is_winner else ""
                    st.write(f"{prefix}{cfg['emoji']} **{emo}** — `{p * 100:.1f}%`")
                    st.progress(float(p))

            # ── Visualizations ──
            st.markdown("### 📊 Acoustic Analysis")
            vcol1, vcol2 = st.columns(2)

            with vcol1:
                fig_wave = plot_waveform_matplotlib(wav, sr)
                st.pyplot(fig_wave, use_container_width=True)
                plt.close(fig_wave)

            with vcol2:
                fig_spec = plot_spectrogram_matplotlib(wav, sr)
                st.pyplot(fig_spec, use_container_width=True)
                plt.close(fig_spec)

            # ── Multi-segment Breakdown (for longer audios) ──
            if mfcc_feat.shape[0] > 1:
                with st.expander("🔍 View Segment-Level Timeline Analysis", expanded=False):
                    st.markdown(
                        f"The audio was sliced into **{mfcc_feat.shape[0]} overlapping segments** (Window: {param_seg_len}s, Hop: {param_seg_len - param_overlap:.2f}s):"
                    )
                    seg_data = []
                    for i, sp in enumerate(seg_probs):
                        seg_pred = EMOTION_CLASSES[int(np.argmax(sp))]
                        seg_data.append({
                            "Segment #": i + 1,
                            "Predicted Emotion": f"{EMOTION_CONFIG[seg_pred]['emoji']} {seg_pred}",
                            "Neutral": f"{sp[0]*100:.1f}%",
                            "Happy": f"{sp[1]*100:.1f}%",
                            "Sad": f"{sp[2]*100:.1f}%",
                            "Angry": f"{sp[3]*100:.1f}%",
                        })
                    st.dataframe(seg_data, use_container_width=True)

        except Exception as err:
            st.error(f"❌ Error during audio processing or inference: {err}")
            st.exception(err)
    else:
        st.info("👆 Please upload an audio file or record your voice using the microphone to begin analysis.")


if __name__ == "__main__":
    main()
