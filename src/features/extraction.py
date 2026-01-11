"""
Feature Extraction for ADHD Detection
Extracts eye-movement biomarkers relevant to ADHD from raw gaze data.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional


class FeatureExtractor:
    """Extract ADHD-relevant features from eye-tracking data."""
    
    def __init__(self, sampling_rate: int = 1000):
        """
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
    
    def compute_velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous velocity in degrees/second.
        
        Args:
            x: Horizontal gaze position (degrees)
            y: Vertical gaze position (degrees)
            
        Returns:
            Velocity array (degrees/second)
        """
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        velocity = np.sqrt(dx**2 + dy**2) / self.dt
        return velocity
    
    def compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute acceleration from velocity."""
        acceleration = np.diff(velocity, prepend=velocity[0]) / self.dt
        return acceleration
    
    def detect_saccades(
        self, 
        velocity: np.ndarray, 
        threshold: float = 30.0
    ) -> List[Tuple[int, int]]:
        """
        Detect saccades using velocity threshold.
        
        Args:
            velocity: Velocity array (degrees/second)
            threshold: Velocity threshold for saccade detection
            
        Returns:
            List of (start, end) indices for each saccade
        """
        above_threshold = velocity > threshold
        
        saccades = []
        in_saccade = False
        start_idx = 0
        
        for i, is_saccade in enumerate(above_threshold):
            if is_saccade and not in_saccade:
                start_idx = i
                in_saccade = True
            elif not is_saccade and in_saccade:
                if i - start_idx >= 2:
                    saccades.append((start_idx, i))
                in_saccade = False
        
        return saccades
    
    def detect_microsaccades(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        amplitude_threshold: float = 1.0
    ) -> List[Tuple[int, int]]:
        """
        Detect microsaccades (tiny involuntary eye movements < 1 degree).
        Higher rate indicates difficulty maintaining fixation (ADHD marker).
        """
        velocity = self.compute_velocity(x, y)
        
        vel_median = np.median(velocity)
        vel_std = np.std(velocity)
        threshold = vel_median + 6 * vel_std
        
        saccades = self.detect_saccades(velocity, threshold=max(threshold, 10))
        
        microsaccades = []
        for start, end in saccades:
            amplitude = np.sqrt(
                (x[end] - x[start])**2 + (y[end] - y[start])**2
            )
            if amplitude < amplitude_threshold:
                microsaccades.append((start, end))
        
        return microsaccades
    
    def compute_bcea(self, x: np.ndarray, y: np.ndarray, p: float = 0.68) -> float:
        """
        Compute Bivariate Contour Ellipse Area (BCEA).
        Measures fixation stability - higher BCEA = more eye "wobble" (ADHD marker).
        
        Args:
            x: Horizontal gaze position
            y: Vertical gaze position
            p: Probability area (default 0.68 for 1 SD)
            
        Returns:
            BCEA in square degrees
        """
        if len(x) < 2:
            return 0.0
        
        std_x = np.std(x)
        std_y = np.std(y)
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        k = -np.log(1 - p)
        bcea = 2 * k * np.pi * std_x * std_y * np.sqrt(1 - correlation**2)
        
        return bcea
    
    def compute_gaze_entropy(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        Compute spatial entropy of gaze distribution.
        Higher entropy = more chaotic/scattered gaze (potential ADHD marker).
        """
        hist, _, _ = np.histogram2d(x, y, bins=n_bins)
        hist = hist.flatten()
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        
        return entropy(hist)
    
    def compute_saccade_latency(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xT: np.ndarray,
        yT: np.ndarray,
        velocity: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Compute reaction time to target changes.
        Measures inhibition control - atypical latency is an ADHD marker.
        """
        if velocity is None:
            velocity = self.compute_velocity(x, y)
        
        target_changes = np.where(
            (np.diff(xT, prepend=xT[0]) != 0) | 
            (np.diff(yT, prepend=yT[0]) != 0)
        )[0]
        
        latencies = []
        threshold = 30.0
        
        for change_idx in target_changes:
            for i in range(change_idx, min(change_idx + 500, len(velocity))):
                if velocity[i] > threshold:
                    latency_ms = (i - change_idx) * 1000 / self.sampling_rate
                    latencies.append(latency_ms)
                    break
        
        return latencies
    
    def extract_all_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all ADHD-relevant features from a gaze DataFrame.
        
        Args:
            df: DataFrame with columns ['n', 'x', 'y', 'val', 'xT', 'yT', 'dP']
            
        Returns:
            Dict of feature names to values
        """
        valid_mask = df['val'] == 0
        x = df.loc[valid_mask, 'x'].values
        y = df.loc[valid_mask, 'y'].values
        
        if len(x) < 100:
            return {}
        
        velocity = self.compute_velocity(x, y)
        acceleration = self.compute_acceleration(velocity)
        
        saccades = self.detect_saccades(velocity)
        microsaccades = self.detect_microsaccades(x, y)
        
        saccade_velocities = []
        saccade_amplitudes = []
        for start, end in saccades:
            saccade_velocities.append(velocity[start:end].max())
            amplitude = np.sqrt((x[end] - x[start])**2 + (y[end] - y[start])**2)
            saccade_amplitudes.append(amplitude)
        
        if 'xT' in df.columns and 'yT' in df.columns:
            xT = df.loc[valid_mask, 'xT'].values
            yT = df.loc[valid_mask, 'yT'].values
            latencies = self.compute_saccade_latency(x, y, xT, yT, velocity)
        else:
            latencies = []
        
        duration_sec = len(x) / self.sampling_rate
        
        features = {
            'velocity_mean': np.mean(velocity),
            'velocity_std': np.std(velocity),
            'velocity_max': np.max(velocity),
            'velocity_median': np.median(velocity),
            
            'acceleration_mean': np.mean(np.abs(acceleration)),
            'acceleration_std': np.std(acceleration),
            
            'saccade_count': len(saccades),
            'saccade_rate': len(saccades) / duration_sec,
            'saccade_velocity_mean': np.mean(saccade_velocities) if saccade_velocities else 0,
            'saccade_amplitude_mean': np.mean(saccade_amplitudes) if saccade_amplitudes else 0,
            
            'microsaccade_count': len(microsaccades),
            'microsaccade_rate': len(microsaccades) / duration_sec,
            
            'bcea': self.compute_bcea(x, y),
            'gaze_entropy': self.compute_gaze_entropy(x, y),
            
            'fixation_x_std': np.std(x),
            'fixation_y_std': np.std(y),
            
            'latency_mean': np.mean(latencies) if latencies else 0,
            'latency_std': np.std(latencies) if len(latencies) > 1 else 0,
            
            'pupil_mean': df.loc[valid_mask, 'dP'].mean() if 'dP' in df.columns else 0,
            'pupil_std': df.loc[valid_mask, 'dP'].std() if 'dP' in df.columns else 0,
            
            'blink_rate': (df['val'] != 0).sum() / duration_sec,
            
            'valid_ratio': valid_mask.mean(),
        }
        
        return features


def extract_adhd_features(
    data: List[Dict],
    sampling_rate: int = 1000,
    tasks: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract features from multiple subjects for ML training.
    
    Args:
        data: List of subject data dicts from GazeBaseLoader
        sampling_rate: Sampling rate in Hz
        tasks: Tasks to extract features from
        
    Returns:
        DataFrame with features for each subject-session-task
    """
    extractor = FeatureExtractor(sampling_rate)
    
    all_features = []
    
    for subject_data in data:
        subject_id = subject_data['subject_id']
        round_num = subject_data['round']
        session = subject_data['session']
        
        for task_code, df in subject_data['data'].items():
            if tasks is not None and task_code not in tasks:
                continue
            
            features = extractor.extract_all_features(df)
            
            if features:
                features['subject_id'] = subject_id
                features['round'] = round_num
                features['session'] = session
                features['task'] = task_code
                all_features.append(features)
    
    return pd.DataFrame(all_features)


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 10000
    
    test_df = pd.DataFrame({
        'n': np.arange(n_samples),
        'x': np.cumsum(np.random.randn(n_samples) * 0.01),
        'y': np.cumsum(np.random.randn(n_samples) * 0.01),
        'val': np.random.choice([0, 0, 0, 0, 1], n_samples),
        'xT': np.zeros(n_samples),
        'yT': np.zeros(n_samples),
        'dP': np.random.randn(n_samples) * 100 + 3000,
    })
    
    extractor = FeatureExtractor(sampling_rate=1000)
    features = extractor.extract_all_features(test_df)
    
    print("Extracted Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
