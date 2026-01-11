# Excite

ADHD detection using eye-tracking and machine learning.

## Overview

This project builds an ML pipeline to detect ADHD based on eye movement patterns. The system uses:

- **GazeBase** dataset for pre-training (learning general gaze dynamics)
- **ADHD-specific datasets** (EEGET_RSOD, TDAH-repository) for fine-tuning
- **Computer Vision** to extract gaze vectors from hardware device
- **Deep Learning** (LSTM/Transformer) for temporal pattern classification

## Pipeline

1. **Data Collection**: Wearable device captures raw eye video
2. **CV Processing**: Extract gaze coordinates, pupil diameter, validity flags
3. **Feature Engineering**: Calculate velocity, fixation jitter, saccade latency
4. **ML Inference**: Trained model outputs ADHD probability

## Key Features Extracted

| Feature | Description | ADHD Biomarker |
|---------|-------------|----------------|
| Saccadic Velocity | Speed of eye jumps | Faster but less accurate |
| Fixation Jitter (BCEA) | Eye wobble during fixation | Higher in ADHD |
| Microsaccade Rate | Tiny involuntary movements | Elevated during focus tasks |
| Saccade Latency | Reaction time to stimulus | Measures inhibition control |

## Project Structure

```
Excite/
├── src/
│   ├── data/           # Data loading (GazeBase)
│   ├── features/       # Feature extraction (velocity, BCEA, etc.)
│   ├── models/         # ML models (Transformer encoder + classifier)
│   ├── training/       # Training pipeline
│   └── inference/      # ADHD detection for your device
├── train.py            # Main training script
├── requirements.txt
└── GazeBase_v2_0/      # Dataset (not committed)
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training

### Stage 1: Pre-train on GazeBase
```bash
# Quick test (1 round, limited subjects)
python train.py pretrain --rounds 1 --max_subjects 10

# Full pre-training
python train.py pretrain --rounds 1 2 3 --epochs 50
```

### Stage 2: Fine-tune on ADHD Data
Requires labeled ADHD data from:
- [EEGET_RSOD](https://github.com/Bing-1997/EEGET_RSOD)
- [TDAH-repository](https://github.com/marcoaaceves/TDAH-repository)

```bash
python train.py finetune --pretrained_path checkpoints/encoder_pretrained.pt
```

## Inference (Your Device)

```python
from src.inference import ADHDDetector

# Load trained model
detector = ADHDDetector('checkpoints/best_classifier.pt')

# Process CSV from your CV model
result = detector.process_csv('gaze_data.csv')
print(f"Prediction: {result['label']}")
print(f"ADHD Probability: {result['adhd_probability']:.2%}")
```

### Real-time Streaming
```python
from src.inference import RealtimeDetector

detector = RealtimeDetector('checkpoints/best_classifier.pt')

# Feed samples from your hardware
for x, y, pupil in stream_from_device():
    result = detector.add_sample(x, y, pupil)
    if result:
        print(f"Prediction: {result['label']}")
```

## CV Model Output Requirements

Your computer vision model must output these parameters per frame:

| Parameter | Description |
|-----------|-------------|
| `x` | Gaze X in Degrees of Visual Angle |
| `y` | Gaze Y in Degrees of Visual Angle |
| `dP` | Pupil diameter (pixels/mm) |
| `val` | Validity (0=good, 1=blink) |

## Data

The GazeBase dataset is not included in this repository due to size. Download from [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) and place in `GazeBase_v2_0/`.

## References

- [GazeBase](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257)
- [EEGET_RSOD (Nature 2025)](https://github.com/Bing-1997/EEGET_RSOD)
- [TDAH-repository](https://github.com/marcoaaceves/TDAH-repository)

## License

MIT