"""
ADHD Detection Inference Module

Use this to run predictions on new eye-tracking data from your device.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gaze_model import GazeEncoder, GazeClassifier
from src.features.extraction import FeatureExtractor


class ADHDDetector:
    """
    ADHD Detection from eye-tracking data.
    
    This is the main interface for your hardware device.
    Feed it gaze data and get ADHD probability.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        sampling_rate: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4
    ):
        """
        Args:
            model_path: Path to trained classifier checkpoint
            device: 'auto', 'cpu', 'cuda', or 'mps'
            sampling_rate: Sampling rate of input data in Hz
            d_model: Model hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() 
                else 'mps' if torch.backends.mps.is_available() 
                else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.sampling_rate = sampling_rate
        self.feature_extractor = FeatureExtractor(sampling_rate)
        
        encoder = GazeEncoder(
            input_dim=5,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        self.model = GazeClassifier(encoder, n_classes=2)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Device: {self.device}")
    
    def _prepare_sequence(self, data: Union[pd.DataFrame, np.ndarray]) -> torch.Tensor:
        """
        Prepare input data for the model.
        
        Expected input format (DataFrame columns or array columns):
            - x: Gaze X in degrees
            - y: Gaze Y in degrees
            - dP: Pupil diameter (optional)
        """
        if isinstance(data, pd.DataFrame):
            x = data['x'].values
            y = data['y'].values
            pupil = data['dP'].values if 'dP' in data.columns else np.zeros_like(x)
        else:
            x = data[:, 0]
            y = data[:, 1]
            pupil = data[:, 2] if data.shape[1] > 2 else np.zeros_like(x)
        
        velocity = self.feature_extractor.compute_velocity(x, y)
        acceleration = self.feature_extractor.compute_acceleration(velocity)
        
        features = np.stack([x, y, velocity, acceleration, pupil], axis=1)
        
        tensor = torch.FloatTensor(features).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """
        Predict ADHD probability from gaze data.
        
        Args:
            data: Eye-tracking data (DataFrame or array)
                  Must have at least 100 samples.
                  
        Returns:
            Dict with:
                - prediction: 0 (Control) or 1 (ADHD)
                - probability: Probability of ADHD
                - confidence: Model confidence (max probability)
        """
        if len(data) < 100:
            raise ValueError("Need at least 100 samples for prediction")
        
        sequence = self._prepare_sequence(data)
        
        with torch.no_grad():
            probs = self.model.predict_proba(sequence)
        
        probs = probs.cpu().numpy()[0]
        prediction = int(np.argmax(probs))
        adhd_prob = float(probs[1])
        confidence = float(np.max(probs))
        
        return {
            'prediction': prediction,
            'label': 'ADHD' if prediction == 1 else 'Control',
            'adhd_probability': adhd_prob,
            'confidence': confidence
        }
    
    def predict_with_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """
        Predict ADHD and also return extracted features.
        Useful for understanding why the model made its prediction.
        """
        result = self.predict(data)
        
        if isinstance(data, pd.DataFrame):
            features = self.feature_extractor.extract_all_features(data)
        else:
            df = pd.DataFrame({
                'n': np.arange(len(data)),
                'x': data[:, 0],
                'y': data[:, 1],
                'val': np.zeros(len(data)),
                'xT': np.zeros(len(data)),
                'yT': np.zeros(len(data)),
                'dP': data[:, 2] if data.shape[1] > 2 else np.zeros(len(data))
            })
            features = self.feature_extractor.extract_all_features(df)
        
        result['features'] = features
        
        return result
    
    def process_csv(self, csv_path: str) -> Dict:
        """
        Process a CSV file from your CV model.
        
        Expected CSV format:
            n,x,y,val,xT,yT,dP
            0,-1.24,0.55,0,0.0,0.0,4520
            1,-1.22,0.54,0,0.0,0.0,4522
            ...
        """
        df = pd.read_csv(csv_path)
        
        required_cols = ['x', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'val' in df.columns:
            df = df[df['val'] == 0]
        
        return self.predict_with_features(df)


class RealtimeDetector:
    """
    Real-time ADHD detection for streaming data from hardware device.
    Accumulates samples and predicts when buffer is full.
    """
    
    def __init__(
        self,
        model_path: str,
        buffer_size: int = 1000,
        overlap: int = 500,
        **kwargs
    ):
        """
        Args:
            model_path: Path to trained model
            buffer_size: Number of samples to accumulate before prediction
            overlap: Number of samples to keep from previous buffer
        """
        self.detector = ADHDDetector(model_path, **kwargs)
        self.buffer_size = buffer_size
        self.overlap = overlap
        
        self.buffer = []
        self.predictions = []
    
    def add_sample(self, x: float, y: float, pupil: float = 0.0) -> Optional[Dict]:
        """
        Add a single sample from the CV model.
        
        Returns prediction dict when buffer is full, None otherwise.
        """
        self.buffer.append([x, y, pupil])
        
        if len(self.buffer) >= self.buffer_size:
            data = np.array(self.buffer)
            result = self.detector.predict(data)
            self.predictions.append(result)
            
            self.buffer = self.buffer[-self.overlap:]
            
            return result
        
        return None
    
    def add_batch(self, samples: np.ndarray) -> list:
        """
        Add multiple samples at once.
        
        Args:
            samples: Array of shape (n_samples, 2) or (n_samples, 3)
            
        Returns:
            List of predictions (may be empty if buffer not full)
        """
        results = []
        
        if samples.shape[1] == 2:
            samples = np.column_stack([samples, np.zeros(len(samples))])
        
        for sample in samples:
            result = self.add_sample(*sample)
            if result is not None:
                results.append(result)
        
        return results
    
    def get_aggregate_prediction(self) -> Optional[Dict]:
        """
        Get aggregate prediction from all accumulated predictions.
        Uses voting/averaging for more robust prediction.
        """
        if not self.predictions:
            return None
        
        adhd_probs = [p['adhd_probability'] for p in self.predictions]
        avg_prob = np.mean(adhd_probs)
        
        adhd_votes = sum(1 for p in self.predictions if p['prediction'] == 1)
        vote_ratio = adhd_votes / len(self.predictions)
        
        return {
            'prediction': 1 if avg_prob > 0.5 else 0,
            'label': 'ADHD' if avg_prob > 0.5 else 'Control',
            'adhd_probability': float(avg_prob),
            'confidence': float(abs(avg_prob - 0.5) * 2),
            'n_predictions': len(self.predictions),
            'vote_ratio': float(vote_ratio)
        }
    
    def reset(self):
        """Clear buffer and predictions."""
        self.buffer = []
        self.predictions = []


if __name__ == "__main__":
    print("ADHD Detector Module")
    print("="*40)
    print("\nUsage example:")
    print("""
    # Single prediction
    detector = ADHDDetector('checkpoints/best_classifier.pt')
    result = detector.process_csv('gaze_data.csv')
    print(f"Prediction: {result['label']}")
    print(f"ADHD Probability: {result['adhd_probability']:.2%}")
    
    # Real-time streaming
    realtime = RealtimeDetector('checkpoints/best_classifier.pt')
    for x, y, pupil in stream_from_device():
        result = realtime.add_sample(x, y, pupil)
        if result:
            print(f"Prediction: {result['label']}")
    
    # Final aggregate
    final = realtime.get_aggregate_prediction()
    print(f"Final: {final['label']} ({final['adhd_probability']:.2%})")
    """)
