"""
ADHD Detection - Inference for Hackathon

Input from CV model (per frame):
- x: Gaze X in degrees (DVA)
- y: Gaze Y in degrees (DVA)  
- dP: Pupil diameter in pixels
- val: Validity flag (0=detected, 1=lost)
- xT, yT: Target position on screen (optional)

Usage:
    detector = ADHDDetector()
    
    # Feed frames one at a time
    for frame in cv_stream:
        result = detector.add_frame(x, y, dP, val)
        if result:
            print(f"ADHD Probability: {result['probability']:.1%}")
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class ADHDModel(nn.Module):
    """Transformer model for ADHD detection."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 64)
        self.pos = nn.Parameter(torch.randn(1, 500, 64) * 0.02)
        enc = nn.TransformerEncoderLayer(64, 4, 128, 0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, 3)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))
    
    def forward(self, x):
        x = self.proj(x) + self.pos
        x = self.transformer(x)
        return self.head(x.mean(dim=1))


class ADHDDetector:
    """ADHD detector using trained ML model."""
    
    def __init__(self, model_path: str = "checkpoints/adhd_model.pt", seq_len: int = 200, fps: int = 30):
        self.seq_len = seq_len
        self.fps = fps
        self.buffer = []
        
        # Load trained model
        if Path(model_path).exists():
            self.model = ADHDModel()
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            acc = checkpoint.get('accuracy', 0)
            print(f"Model loaded: {model_path} (accuracy: {acc:.1%})")
        else:
            self.model_loaded = False
            print(f"WARNING: Model not found at {model_path}")
            print("Running in DEMO mode")
    
    def add_frame(self, x: float, y: float, dP: float, val: int, xT: float = 0, yT: float = 0) -> dict:
        """
        Add a single frame from CV model.
        
        Args:
            x: Gaze X position (degrees)
            y: Gaze Y position (degrees)
            dP: Pupil diameter (pixels)
            val: Validity flag (0=valid, 1=invalid)
            xT: Target X position (optional)
            yT: Target Y position (optional)
            
        Returns:
            dict with prediction if buffer full, else None
        """
        # Skip invalid frames
        if val == 1:
            return None
        
        # Store frame data [x, y, dP]
        self.buffer.append([x, y, dP])
        
        # Make prediction when buffer is full
        if len(self.buffer) >= self.seq_len:
            result = self._predict()
            # Slide buffer by half
            self.buffer = self.buffer[self.seq_len // 2:]
            return result
        
        return None
    
    def _predict(self) -> dict:
        """Run prediction using trained ML model."""
        seq = np.array(self.buffer[-self.seq_len:], dtype=np.float32)
        
        # Pad to 500 if needed (model expects 500)
        if len(seq) < 500:
            pad_len = 500 - len(seq)
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
        
        x, y, pupil = seq[:, 0], seq[:, 1], seq[:, 2]
        
        # Compute velocity and acceleration (same as training)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        velocity = np.sqrt(dx**2 + dy**2) * self.fps
        acceleration = np.abs(np.diff(velocity, prepend=velocity[0])) * self.fps
        
        # Stack features: [x, y, velocity, acceleration, pupil]
        features = np.stack([x, y, velocity, acceleration, pupil], axis=1)
        
        # Normalize each feature
        for i in range(5):
            std = features[:, i].std()
            if std > 1e-6:
                features[:, i] = (features[:, i] - features[:, i].mean()) / std
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not self.model_loaded:
            # Demo mode
            return {
                'probability': 0.5,
                'prediction': 'Unknown',
                'confidence': 0.0
            }
        
        # Run ML model
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features).unsqueeze(0)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            adhd_prob = probs[0, 1].item()
        
        return {
            'probability': adhd_prob,
            'prediction': 'ADHD' if adhd_prob > 0.5 else 'Control',
            'confidence': abs(adhd_prob - 0.5) * 2
        }
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = []


# Demo usage
if __name__ == "__main__":
    print("="*50)
    print("ADHD DETECTOR - HACKATHON DEMO")
    print("="*50)
    
    detector = ADHDDetector()
    
    # Simulate CV input
    print("\nSimulating 600 frames of eye tracking data...")
    for i in range(600):
        # Fake data - replace with real CV output
        x = np.random.randn() * 2  # Gaze X (degrees)
        y = np.random.randn() * 2  # Gaze Y (degrees)
        dP = 50 + np.random.randn() * 5  # Pupil diameter
        val = 0  # Valid frame
        
        result = detector.add_frame(x, y, dP, val)
        
        if result:
            print(f"\n>>> Frame {i}: {result['prediction']}")
            print(f"    Probability: {result['probability']:.1%}")
            print(f"    Confidence: {result['confidence']:.1%}")
    
    print("\n" + "="*50)
    print("Integration with your CV code:")
    print("="*50)
    print("""
from inference import ADHDDetector

detector = ADHDDetector()

# In your OpenCV loop:
while True:
    frame = camera.read()
    x, y, dP, val, xT, yT = your_eye_tracker(frame)
    
    result = detector.add_frame(x, y, dP, val, xT, yT)
    if result:
        print(f"ADHD: {result['probability']:.0%}")
""")
