"""
ADHD Prediction - Connect CV Model Output to ML Model

Usage:
    # From CSV file (CV model output)
    python run_prediction.py --input eye_tracking_data.csv
    
    # From JSON file
    python run_prediction.py --input eye_tracking_data.json

Expected CV output format (per frame):
    x     - Gaze X in degrees (DVA)
    y     - Gaze Y in degrees (DVA)
    dP    - Pupil diameter in pixels
    val   - Validity flag (0=detected, 1=lost)
    xT    - Target X position (optional)
    yT    - Target Y position (optional)
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from inference import ADHDDetector


def load_cv_output(filepath: str) -> pd.DataFrame:
    """Load CV model output from CSV or JSON."""
    path = Path(filepath)
    
    if path.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif path.suffix == '.json':
        with open(filepath) as f:
            data = json.load(f)
        # Handle both list of dicts and dict of lists
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .csv or .json")
    
    # Normalize column names (handle variations)
    col_map = {
        'gaze_x': 'x', 'gazeX': 'x', 'GazeX': 'x', 'X': 'x',
        'gaze_y': 'y', 'gazeY': 'y', 'GazeY': 'y', 'Y': 'y',
        'pupil': 'dP', 'pupil_diameter': 'dP', 'PupilDiameter': 'dP', 'diameter': 'dP',
        'valid': 'val', 'validity': 'val', 'Validity': 'val',
        'target_x': 'xT', 'targetX': 'xT', 'TargetX': 'xT',
        'target_y': 'yT', 'targetY': 'yT', 'TargetY': 'yT',
    }
    df = df.rename(columns=col_map)
    
    # Check required columns
    required = ['x', 'y']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")
    
    # Fill missing optional columns
    if 'dP' not in df.columns:
        df['dP'] = 0.0
    if 'val' not in df.columns:
        df['val'] = 0
    
    return df


def run_prediction(filepath: str, output_path: str = None) -> dict:
    """
    Run ADHD prediction on CV model output.
    
    Args:
        filepath: Path to CV output file (CSV or JSON)
        output_path: Optional path to save results
        
    Returns:
        dict with predictions
    """
    print("="*50)
    print("ADHD PREDICTION FROM VIDEO")
    print("="*50)
    
    # Load CV output
    print(f"\nLoading: {filepath}")
    df = load_cv_output(filepath)
    print(f"Frames: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize detector
    detector = ADHDDetector()
    
    # Process frames
    print(f"\nProcessing {len(df)} frames...")
    predictions = []
    
    for i, row in df.iterrows():
        x = float(row['x'])
        y = float(row['y'])
        dP = float(row.get('dP', 0))
        val = int(row.get('val', 0))
        xT = float(row.get('xT', 0))
        yT = float(row.get('yT', 0))
        
        result = detector.add_frame(x, y, dP, val, xT, yT)
        
        if result:
            predictions.append({
                'frame': i,
                'probability': result['probability'],
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
            print(f"  Frame {i}: {result['prediction']} ({result['probability']:.1%})")
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    if predictions:
        probs = [p['probability'] for p in predictions]
        avg_prob = np.mean(probs)
        final_pred = 'ADHD' if avg_prob > 0.5 else 'Control'
        
        result = {
            'final_prediction': final_pred,
            'average_probability': float(avg_prob),
            'confidence': float(abs(avg_prob - 0.5) * 2),
            'num_predictions': len(predictions),
            'all_predictions': predictions
        }
        
        print(f"\nFinal Prediction: {final_pred}")
        print(f"ADHD Probability: {avg_prob:.1%}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Based on: {len(predictions)} prediction windows")
    else:
        result = {
            'error': 'Not enough frames for prediction (need 500+)',
            'frames_provided': len(df)
        }
        print(f"\nError: Need at least 500 frames, got {len(df)}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADHD prediction on CV model output")
    parser.add_argument("--input", "-i", required=True, help="Path to CV output (CSV or JSON)")
    parser.add_argument("--output", "-o", help="Path to save results (JSON)")
    args = parser.parse_args()
    
    run_prediction(args.input, args.output)
