"""
ADHD Dataset Loader

Loads pre-extracted ADHD eye-tracking sequences from .npz file.
Format matches pre-training: [x, y, pupil] sequences.
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def load_adhd_sequences(npz_path: str = "data/adhd/adhd_sequences.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted ADHD sequences.
    
    Args:
        npz_path: Path to .npz file with 'sequences' and 'labels'
        
    Returns:
        sequences: (n_samples, seq_len, 3) array with [x, y, pupil]
        labels: (n_samples,) array with 0=Control, 1=ADHD
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ADHD data not found: {path}\n"
            f"Run: python scripts/extract_adhd_data.py"
        )
    
    data = np.load(npz_path)
    sequences = data['sequences'].astype(np.float32)
    labels = data['labels'].astype(np.int64)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"  ADHD: {np.sum(labels == 1)}, Control: {np.sum(labels == 0)}")
    
    return sequences, labels
