import os
import math
import pickle
import random
import json
from collections import Counter
import numpy as np
import librosa
from tqdm import tqdm
from datasets import load_dataset
from python_speech_features import logfbank, fbank, sigproc


# 4-class SER mapping (standard IEMOCAP benchmark)
# Neutral: 0, Sad: 1, Angry: 2, Happy/Excited: 3
EMOTION_MAP = {
    'neu': 0, 'neutral': 0, '01': 0,
    'sad': 1, 'sadness': 1, '04': 1,
    'ang': 2, 'angry': 2, '05': 2,
    'hap': 3, 'happy': 3, '03': 3,
    'exc': 3, 'excited': 3, '07': 3,
}


class FeatureExtractor:
    """Extract acoustic features compatible with CTMAM model input."""
    def __init__(self, sample_rate=16000, nmfcc=26):
        self.sample_rate = sample_rate
        self.nmfcc = nmfcc

    def get_features(self, features_to_use, X):
        if features_to_use == 'mfcc':
            return self.get_mfcc(X, self.nmfcc)
        elif features_to_use == 'logfbank':
            return self.get_logfbank(X)
        elif features_to_use == 'fbank':
            return self.get_fbank(X)
        elif features_to_use == 'melspectrogram':
            return self.get_melspectrogram(X)
        elif features_to_use == 'spectrogram':
            return self.get_spectrogram(X)
        else:
            raise NotImplementedError(f"Feature {features_to_use} not supported")

    def get_mfcc(self, X, nmfcc=26):
        def _get_mfcc(x):
            return librosa.feature.mfcc(y=x, sr=self.sample_rate, n_mfcc=nmfcc)
        return np.apply_along_axis(_get_mfcc, 1, X)

    def get_logfbank(self, X):
        def _get_logfbank(x):
            return logfbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000, nfilt=40)
        return np.apply_along_axis(_get_logfbank, 1, X)

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out
        return np.apply_along_axis(_get_fbank, 1, X)

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.sample_rate, n_fft=800, hop_length=400)[np.newaxis, :]
            return np.log10(np.maximum(mel, 1e-10)).squeeze()
        return np.apply_along_axis(_get_melspectrogram, 1, X)

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]
        return np.apply_along_axis(_get_spectrogram, 1, X)


def segment_audio(wav_data, sample_rate=16000, segment_length=1.8, overlap=1.6, padding=True):
    """Slice 1D audio waveform into fixed-length overlapping windows."""
    seg_wav_len = int(segment_length * sample_rate)
    wav_len = len(wav_data)

    if seg_wav_len > wav_len:
        if padding:
            n = math.ceil(seg_wav_len / wav_len)
            wav_data = np.hstack(n * [wav_data])
            wav_len = len(wav_data)
        else:
            return None

    X = []
    hop_samples = int((segment_length - overlap) * sample_rate)
    if hop_samples <= 0:
        hop_samples = int(0.2 * sample_rate)

    index = 0
    while index + seg_wav_len <= wav_len:
        X.append(wav_data[int(index): int(index + seg_wav_len)])
        index += hop_samples

    if len(X) == 0:
        return None
    return np.array(X)


def load_and_process_hf_iemocap(
    dataset_name="AbstractTTS/IEMOCAP",
    split="train",
    datadir="data/IEMOCAP",
    features_to_use="mfcc",
    sample_rate=16000,
    nmfcc=26,
    segment_length=1.8,
    train_overlap=1.6,
    test_overlap=1.6,
    split_rate=0.8,
    save_features=True,
    output_filename=None,
):
    """
    Downloads/loads IEMOCAP from Hugging Face, preprocesses the audio waveforms,
    extracts CTMAM-compatible features, and saves them to the expected pickle cache.
    """
    print(f"[*] Loading dataset '{dataset_name}' from Hugging Face...")
    ds = load_dataset(dataset_name)

    # Handle dataset dict vs single split
    if isinstance(ds, dict):
        if split in ds:
            dataset = ds[split]
        else:
            # Concatenate or use first split
            first_key = list(ds.keys())[0]
            dataset = ds[first_key]
    else:
        dataset = ds

    print(f"[*] Total records downloaded: {len(dataset)}")

    feature_extractor = FeatureExtractor(sample_rate=sample_rate, nmfcc=nmfcc)
    os.makedirs(datadir, exist_ok=True)

    if output_filename is None:
        output_filename = os.path.join(datadir, f"features_{features_to_use}_all.pkl")

    # Parse records and map emotion labels
    print("[*] Parsing audio and labels...")
    valid_samples = []
    skipped = 0

    for item in tqdm(dataset, desc="Reading items"):
        # Resolve emotion label column
        label_raw = None
        for key in ["major_emotion", "emotion", "label", "emotion_label"]:
            if key in item and item[key] is not None:
                label_raw = str(item[key]).strip().lower()
                break

        if label_raw is None or label_raw not in EMOTION_MAP:
            skipped += 1
            continue

        label_id = EMOTION_MAP[label_raw]

        # Extract audio waveform
        audio_entry = item.get("audio", None)
        if audio_entry is None:
            continue

        if isinstance(audio_entry, dict):
            wav_data = np.array(audio_entry["array"], dtype=np.float32)
            sr = audio_entry.get("sampling_rate", sample_rate)
        else:
            continue

        # Resample if sample rate doesn't match
        if sr != sample_rate:
            wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=sample_rate)

        valid_samples.append((wav_data, label_id))

    print(f"[*] Kept {len(valid_samples)} samples (filtered out {skipped} non-target/unlabeled samples)")

    # Split into train and validation sets
    n = len(valid_samples)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    train_size = int(n * split_rate)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_files = [valid_samples[i] for i in train_indices]
    valid_files = [valid_samples[i] for i in valid_indices]

    print(f"[*] Training samples: {len(train_files)}, Validation samples: {len(valid_files)}")

    # Extract training segments & features
    print("[*] Extracting features for training set...")
    train_X, train_y = [], []
    for wav_data, label in tqdm(train_files, desc="Processing Train"):
        segments = segment_audio(
            wav_data,
            sample_rate=sample_rate,
            segment_length=segment_length,
            overlap=train_overlap,
            padding=True,
        )
        if segments is None:
            continue
        feats = feature_extractor.get_features(features_to_use, segments)
        train_X.append(feats)
        train_y.extend([label] * len(feats))

    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    print(f"[*] Train segments shape: {train_X.shape}, labels distribution: {Counter(train_y)}")

    # Extract validation segments & features
    print("[*] Extracting features for validation set...")
    val_dict = []
    val_y = []
    for wav_data, label in tqdm(valid_files, desc="Processing Val"):
        segments = segment_audio(
            wav_data,
            sample_rate=sample_rate,
            segment_length=segment_length,
            overlap=test_overlap,
            padding=True,
        )
        if segments is None:
            continue
        feats = feature_extractor.get_features(features_to_use, segments)
        val_dict.append({"X": feats, "y": label})
        val_y.append(label)

    print(f"[*] Validation utterances: {len(val_dict)}, labels distribution: {Counter(val_y)}")

    info = {
        "train": str(Counter(train_y)),
        "test": str(Counter(val_y)),
        "dataset_source": dataset_name,
        "features_to_use": features_to_use,
    }

    if save_features:
        print(f"[*] Saving extracted features to '{output_filename}'...")
        with open(output_filename, "wb") as f:
            pickle.dump(
                {
                    "train_X": train_X,
                    "train_y": train_y,
                    "val_dict": val_dict,
                    "info": info,
                },
                f,
            )
        print(f"[✓] Feature preprocessing complete! You can now run:")
        print(f"    python train_IEMOCAP.py -f {features_to_use} -m CTMAM -b 128 -e 150")

    return train_X, train_y, val_dict, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hugging Face IEMOCAP Dataset Loader for CTMAM")
    parser.add_argument("--dataset_name", default="AbstractTTS/IEMOCAP", type=str, help="Hugging Face dataset repo name")
    parser.add_argument("-f", "--features_to_use", default="mfcc", type=str, choices=["mfcc", "logfbank", "fbank", "spectrogram", "melspectrogram"])
    parser.add_argument("-s", "--sample_rate", default=16000, type=int)
    parser.add_argument("-n", "--nmfcc", default=26, type=int)
    parser.add_argument("--segment_length", default=1.8, type=float)
    parser.add_argument("--split_rate", default=0.8, type=float)
    parser.add_argument("-d", "--datadir", default="data/IEMOCAP", type=str)

    args = parser.parse_args()

    load_and_process_hf_iemocap(
        dataset_name=args.dataset_name,
        datadir=args.datadir,
        features_to_use=args.features_to_use,
        sample_rate=args.sample_rate,
        nmfcc=args.nmfcc,
        segment_length=args.segment_length,
        split_rate=args.split_rate,
    )