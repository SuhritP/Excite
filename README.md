<p align="center">
  <img src="https://img.shields.io/badge/EXCITE-Eye%20Tracking%20ADHD%20Detection-8B5CF6?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjMiLz48cGF0aCBkPSJNMiAxMnMzLTcgMTAtNyAxMCA3IDEwIDctMy43IDctMTAgNy0xMC03LTEwLTdaIi8+PC9zdmc+" alt="EXCITE">
</p>

<h1 align="center">EXCITE</h1>
<h3 align="center">Eye-tracking Classification for Intelligent Therapeutic Evaluation</h3>

<p align="center">
  <strong>AI-powered ADHD screening through eye movement analysis</strong><br>
  Screen in minutes, not months.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-2.0+-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
</p>

---

## 🧠 About

**EXCITE** is pioneering a new frontier in ADHD detection. Our mission is to develop an accessible, non-invasive screening tool that can identify ADHD indicators through eye movement analysis.

Traditional ADHD diagnosis relies on subjective behavioral assessments and lengthy clinical evaluations. EXCITE offers a different approach—**using the eyes as a window into cognitive function**. By analyzing how individuals track visual stimuli, our system can detect patterns associated with attention disorders in minutes, not months.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Non-Invasive Screening** | No blood tests, no brain scans. Simply watch dots on a screen. |
| ⚡ **Rapid Results** | Complete screening in under 5 minutes with instant AI analysis. |
| 📦 **Portable Hardware** | Runs on Raspberry Pi—deploy anywhere for ~$50. |
| 🧠 **Research-Backed** | Trained on 12,000+ eye-tracking recordings from GazeBase. |
| 👁 **Real-Time Visualization** | Watch gaze patterns as the AI analyzes eye movements. |
| 🔒 **Privacy-First** | All processing happens locally. Data never leaves the device. |

## 🏗️ Architecture

```
+------------------------------------------------------------------+
|                         EXCITE Pipeline                          |
+------------------------------------------------------------------+
|                                                                  |
|   Camera           CV Model            ML Model                  |
|   --------  --->   -----------  --->   -----------  --->  Result |
|   Eye Video        Gaze Extraction     ADHD Detection            |
|                    (eyetrack.py)       (inference.py)            |
|                                                                  |
|   Extracts:                 Analyzes:                            |
|   - Pupil position          - Saccade velocity                   |
|   - Pupil diameter          - Fixation stability                 |
|   - Gaze coordinates        - Gaze entropy                       |
|   - Validity flags          - Temporal patterns                  |
|                                                                  |
+------------------------------------------------------------------+
```

## 📁 Project Structure

```
Excite/
├── webapp.py               # 🌐 Web application (Flask + beautiful UI)
├── app.py                  # 📱 Basic web interface
├── eyetrack.py             # 👁 Computer vision pipeline (pupil detection)
├── inference.py            # 🧠 ADHD detection model
├── train.py                # 🏋️ Model training script
├── run_prediction.py       # 🔮 Batch prediction utility
│
├── src/                    # Core modules
│   ├── data/               #   Data loading (GazeBase, ADHD datasets)
│   ├── features/           #   Feature extraction (velocity, BCEA, etc.)
│   ├── models/             #   ML models (Transformer encoder)
│   ├── training/           #   Training pipeline
│   └── inference/          #   Real-time detection
│
├── checkpoints/            # 🎯 Trained model weights
│   ├── adhd_model.pt       #   ADHD classifier
│   ├── encoder_pretrained.pt   Pre-trained encoder
│   └── best_classifier.pt  #   Best performing model
│
├── config/                 # ⚙️ Configuration files
├── scripts/                # 🔧 Training scripts (GPU cluster)
├── data/                   # 📊 ADHD training sequences
├── uploads/                # 📤 User uploaded videos
└── outputs/                # 📁 Processing results
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SuhritP/Excite.git
cd Excite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
python webapp.py
# Open http://localhost:5000 in your browser
```

### Command Line Usage

```python
from inference import ADHDDetector

# Initialize detector
detector = ADHDDetector()

# Feed eye tracking data frame by frame
for frame in video_frames:
    x, y, pupil_diameter = extract_gaze(frame)
    result = detector.add_frame(x, y, pupil_diameter, validity=0)
    
    if result:
        print(f"ADHD Probability: {result['probability']:.1%}")
        print(f"Prediction: {result['prediction']}")
```

## 🔬 Technology

### Computer Vision Pipeline

Our custom CV system extracts eye movement features in real-time:

- **Pupil Detection**: Dark-region search with ellipse fitting
- **Gaze Estimation**: Maps pupil position to visual angle
- **Glint Tracking**: Corneal reflection for calibration
- **Blink Detection**: Validity flagging for data quality

### Transformer Neural Network

At the heart of EXCITE is a state-of-the-art **Transformer encoder**:

```python
ADHDModel(
  (proj): Linear(5, 64)           # Input projection
  (transformer): TransformerEncoder(
    (layers): 3x TransformerEncoderLayer(
      d_model=64, nhead=4, dim_feedforward=128
    )
  )
  (head): Linear(64, 2)           # Classification head
)
```

### Two-Stage Training

1. **Pre-training**: Learn general gaze dynamics from GazeBase (12,000+ recordings)
2. **Fine-tuning**: Specialize on ADHD detection with labeled clinical data

### ADHD-Relevant Biomarkers

| Biomarker | Description | ADHD Pattern |
|-----------|-------------|--------------|
| Saccadic Velocity | Speed of eye jumps | Faster but less accurate |
| Fixation Stability (BCEA) | Eye wobble during fixation | Higher variability |
| Microsaccade Rate | Tiny involuntary movements | Elevated during tasks |
| Gaze Entropy | Randomness of gaze patterns | Higher unpredictability |
| Pupil Dynamics | Diameter changes | Altered responses |

## 📊 Datasets

### GazeBase Data Repository
> Griffith, H., Lohr, D., Abdulin, E., & Komogortsev, O. (2020)
> 
> Large-scale, multi-stimulus, longitudinal eye movement dataset.
> 
> [https://doi.org/10.6084/m9.figshare.12912257](https://doi.org/10.6084/m9.figshare.12912257)

### ADHD Pupil Size Dataset
> Krejtz, K., et al. (2018)
> 
> Pupil size dataset for ADHD research.
> 
> [https://doi.org/10.6084/m9.figshare.7218725](https://doi.org/10.6084/m9.figshare.7218725)

## 🖥️ CV Model Output Format

Your eye tracker must provide these values per frame:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | float | Gaze X position (degrees visual angle) |
| `y` | float | Gaze Y position (degrees visual angle) |
| `dP` | float | Pupil diameter (pixels or mm) |
| `val` | int | Validity flag (0=valid, 1=blink/lost) |

## 🏋️ Training

### Pre-train on GazeBase

```bash
# Quick test
python train.py pretrain --rounds 1 --max_subjects 10

# Full training
python train.py pretrain --rounds 1 2 3 --epochs 50
```

### Fine-tune on ADHD Data

```bash
python train.py finetune --pretrained_path checkpoints/encoder_pretrained.pt
```

### GPU Cluster Scripts

```bash
# Setup environment
./scripts/setup_cluster.sh

# Run training
./scripts/train_full.sh
```

## ⚠️ Disclaimer

**EXCITE is a screening tool designed to complement—not replace—professional clinical evaluation for ADHD.**

This software is for research and educational purposes. Always consult qualified healthcare professionals for medical diagnoses.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for the future of cognitive diagnostics
</p>
