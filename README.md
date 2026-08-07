# SER-CNN-Transformer
# Speech Emotion Recognition — CNN-Transformer + Multidimensional Attention
This repository implements a Speech Emotion Recognition (SER) system based on a hybrid
CNN-Transformer architecture enhanced with a Multidimensional Attention Mechanism.

## Highlights
- Hybrid CNN + Transformer backbone for robust speech representation
- Multidimensional Attention Mechanism for improved emotion discrimination
- Scripts for preprocessing, training, and inference on IEMOCAP features

## Requirements
```bash
pip install -r requirements.txt
```

## Quick Start

1. Preprocess IEMOCAP (optional if using built-in loader):
```bash
python preprocessing/process_IEMOCAP.py
```

2. Train a model with MFCC features (example):
```bash
python train_IEMOCAP.py -f mfcc -m CTMAM -b 128 -e 150 -l 0.001 -g 0
```

3. Run inference on an audio file (feature extraction + predict):
```bash
python predict.py --model-path data/IEMOCAP/model_CTMAM_mfcc_all.pth --file examples/sample.wav
```

Notes:
- Use `-g` to select GPU id (set to `-1` for CPU).
- Use `-f` to change feature type (e.g., `mfcc`, `fbank` if supported).

## Data
- The repository contains helper scripts in `preprocessing/` to prepare the IEMOCAP features.
- Example dataset folder: `data/IEMOCAP/` with precomputed feature files and checkpoints.

## Project Structure
- `train_IEMOCAP.py` — training entrypoint
- `predict.py` — inference script
- `models.py` — model definitions (CNN-Transformer and attention modules)
- `data_loader.py` — dataset and dataloader utilities
- `preprocessing/` — data processing helpers for IEMOCAP

## Pretrained Models & Checkpoints
- Example checkpoint included: `data/IEMOCAP/model_CTMAM_mfcc_all.pth` and metric/loss logs in the same folder.
- To evaluate or resume training, point `--model-path` to the checkpoint file.

## Evaluation
- Training script saves checkpoints and logs final metrics to `data/IEMOCAP/`.
- Add evaluation code or use `predict.py` to compute per-file predictions and aggregate metrics.


## License
MIT

