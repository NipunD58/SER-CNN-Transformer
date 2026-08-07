
"""
Speech Emotion Recognition (SER) Prediction Script using trained CTMAM Model.
Usage:
    python predict.py "path/to/your_audio.wav"
    python predict.py --audio "path/to/your_audio.wav"
    python predict.py   (will prompt for audio path interactively)
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import librosa

import models

# 4-class SER Emotion Labels for IEMOCAP
EMOTION_CLASSES = ['Neutral', 'Happy', 'Sad', 'Angry']
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'IEMOCAP', 'model_CTMAM_mfcc_all.pth'
)


def extract_mfcc_features(wav_data, sample_rate=16000, nmfcc=26, segment_length=1.8, overlap=1.6):
    """
    Slice audio waveform into 1.8-second overlapping windows (hop 0.2s)
    and extract 26 MFCC coefficients matching input shape (N, 26, 57).
    """
    seg_wav_len = int(segment_length * sample_rate)
    wav_len = len(wav_data)

    # Pad if audio is shorter than 1.8s
    if seg_wav_len > wav_len:
        n = math.ceil(seg_wav_len / max(wav_len, 1))
        wav_data = np.hstack(n * [wav_data])
        wav_len = len(wav_data)

    # Slice into segments
    segments = []
    hop_samples = int((segment_length - overlap) * sample_rate)
    if hop_samples <= 0:
        hop_samples = int(0.2 * sample_rate)

    index = 0
    while index + seg_wav_len <= wav_len:
        segments.append(wav_data[int(index): int(index + seg_wav_len)])
        index += hop_samples

    if len(segments) == 0:
        segments.append(wav_data[:seg_wav_len])

    segments = np.array(segments)

    # Extract 26-dim MFCC per segment -> shape: (num_segments, 26, 57)
    def _get_mfcc(x):
        return librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=nmfcc)

    mfcc_features = np.apply_along_axis(_get_mfcc, 1, segments)
    return mfcc_features


def load_trained_model(model_path=DEFAULT_MODEL_PATH, shape=(26, 57), device='cpu'):
    """
    Instantiate CTMAM model and load checkpoint weights.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    model = models.CTMAM(shape=shape)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_emotion(audio_path, model_path=DEFAULT_MODEL_PATH, device=None):
    """
    Predict the emotion of a given audio file path.
    
    Returns:
        pred_label (str): Predicted emotion ('Neutral', 'Happy', 'Sad', 'Angry')
        confidence (float): Confidence score (0.0 to 1.0)
        prob_dict (dict): Probability breakdown for all emotion classes
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load and resample audio to 16kHz
    wav, sr = librosa.load(audio_path, sr=16000)

    # 2. Extract segment MFCC features (N, 26, 57)
    features = extract_mfcc_features(wav, sample_rate=16000, nmfcc=26)

    # 3. Load model
    model = load_trained_model(model_path=model_path, shape=(26, 57), device=device)

    # 4. Prepare tensor input
    x = torch.from_numpy(features).float().to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = torch.cat((x, x), 0)

    # 5. Forward inference
    with torch.no_grad():
        out = model(x.unsqueeze(1))  # (num_segments, 4)
        probs = F.softmax(out, dim=-1).mean(dim=0).cpu().numpy()  # Average probabilities across segments

    pred_class_idx = int(np.argmax(probs))
    pred_label = EMOTION_CLASSES[pred_class_idx]
    confidence = float(probs[pred_class_idx])
    prob_dict = {emotion: float(prob) for emotion, prob in zip(EMOTION_CLASSES, probs)}

    return pred_label, confidence, prob_dict


def display_results(audio_path, pred_label, confidence, prob_dict):
    """Format and print prediction outputs with visual bar chart."""
    print("\n" + "=" * 55)
    print("         SPEECH EMOTION RECOGNITION RESULT")
    print("=" * 55)
    print(f"  Audio File:        {os.path.basename(audio_path)}")
    print(f"  Full Path:         {os.path.abspath(audio_path)}")
    print(f"  Predicted Emotion: {pred_label.upper()}")
    print(f"  Confidence:        {confidence * 100:.2f}%")
    print("-" * 55)
    print("  Probability Breakdown:")
    for emotion, prob in prob_dict.items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        marker = " <== [PREDICTION]" if emotion.lower() == pred_label.lower() else ""
        print(f"    {emotion:8s} : [{bar}] {prob * 100:6.2f}%{marker}")
    print("=" * 55 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Speech Emotion from Audio File using CTMAM")
    parser.add_argument('audio_file', nargs='?', default=None, help="Path to input audio file (.wav, .mp3, .flac, etc.)")
    parser.add_argument('-a', '--audio', type=str, default=None, help="Path to audio file (alternative flag)")
    parser.add_argument('-m', '--model_path', type=str, default=DEFAULT_MODEL_PATH, help="Path to .pth model weights")
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU device ID (-1 for CPU)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Determine input audio path from positional arg or --audio flag
    audio_path = args.audio_file or args.audio

    # If no audio provided via CLI, prompt interactively
    if not audio_path:
        print("=" * 55)
        print("  CTMAM Speech Emotion Recognition Predictor")
        print("=" * 55)
        audio_path = input("Enter the path to your audio file (.wav, .mp3, etc.): ").strip()
        # Strip surrounding quotes if dragged & dropped into terminal
        audio_path = audio_path.strip('\'"')

    if not audio_path:
        print("[!] No audio file specified. Exiting.")
        sys.exit(1)

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    try:
        pred_label, confidence, prob_dict = predict_emotion(
            audio_path=audio_path,
            model_path=args.model_path,
            device=device
        )
        display_results(audio_path, pred_label, confidence, prob_dict)
    except Exception as e:
        print(f"\n[Error] Failed to process audio file: {e}")
        sys.exit(1)
