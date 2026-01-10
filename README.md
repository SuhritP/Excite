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

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data

The GazeBase dataset is not included in this repository due to size. Download from [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) and place in `GazeBase_v2_0/`.

## License

MIT