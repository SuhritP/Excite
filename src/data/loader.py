"""
GazeBase Data Loader
Handles extraction and loading of GazeBase dataset CSVs.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class GazeBaseLoader:
    """Load and manage GazeBase eye-tracking dataset."""
    
    TASK_CODES = {
        'FXS': 'Fixations',
        'HSS': 'Horizontal_Saccades', 
        'RAN': 'Random_Saccades',
        'TEX': 'Reading',
        'VD1': 'Video_1',
        'VD2': 'Video_2',
        'BLG': 'Balura_Game'
    }
    
    COLUMNS = ['n', 'x', 'y', 'val', 'xT', 'yT', 'dP', 'lab']
    
    def __init__(self, data_root: str, sampling_rate: int = 1000):
        """
        Args:
            data_root: Path to GazeBase_v2_0 directory
            sampling_rate: Original sampling rate in Hz (GazeBase is 1000Hz)
        """
        self.data_root = Path(data_root)
        self.sampling_rate = sampling_rate
        self._validate_root()
        
    def _validate_root(self):
        """Check if data root exists and has expected structure."""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        rounds = list(self.data_root.glob("Round_*"))
        if not rounds:
            raise ValueError(f"No Round_* directories found in {self.data_root}")
        print(f"Found {len(rounds)} rounds in dataset")
    
    def extract_all_zips(self, rounds: Optional[List[int]] = None):
        """
        Extract all subject zip files.
        
        Args:
            rounds: List of round numbers to extract (e.g., [1, 2]). None = all.
        """
        if rounds is None:
            round_dirs = sorted(self.data_root.glob("Round_*"))
        else:
            round_dirs = [self.data_root / f"Round_{r}" for r in rounds]
        
        for round_dir in round_dirs:
            if not round_dir.exists():
                print(f"Skipping {round_dir.name} - not found")
                continue
                
            zip_files = list(round_dir.glob("Subject_*.zip"))
            print(f"\n{round_dir.name}: {len(zip_files)} zip files to extract")
            
            for zip_path in tqdm(zip_files, desc=f"Extracting {round_dir.name}"):
                subject_dir = zip_path.with_suffix('')
                
                if subject_dir.exists():
                    continue
                    
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(round_dir)
                except Exception as e:
                    print(f"Error extracting {zip_path.name}: {e}")
    
    def get_subject_ids(self, round_num: int) -> List[str]:
        """Get list of subject IDs for a given round."""
        round_dir = self.data_root / f"Round_{round_num}"
        subjects = []
        
        for item in round_dir.iterdir():
            if item.is_dir() and item.name.startswith("Subject_"):
                subjects.append(item.name.replace("Subject_", ""))
        
        return sorted(subjects)
    
    def load_subject_session(
        self, 
        subject_id: str, 
        round_num: int, 
        session: int,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all task data for a subject session.
        
        Args:
            subject_id: Subject ID (e.g., "1001")
            round_num: Round number (1-9)
            session: Session number (1 or 2)
            tasks: List of task codes to load (e.g., ['FXS', 'RAN']). None = all.
            
        Returns:
            Dict mapping task code to DataFrame
        """
        round_dir = self.data_root / f"Round_{round_num}"
        subject_dir = round_dir / f"Subject_{subject_id}"
        session_dir = subject_dir / f"S{session}"
        
        if not session_dir.exists():
            return {}
        
        data = {}
        task_dirs = list(session_dir.glob("S*_*/"))
        
        for task_dir in task_dirs:
            task_name = task_dir.name.split('_', 1)[1] if '_' in task_dir.name else task_dir.name
            task_code = None
            
            for code, name in self.TASK_CODES.items():
                if name == task_name:
                    task_code = code
                    break
            
            if tasks is not None and task_code not in tasks:
                continue
            
            csv_files = list(task_dir.glob("*.csv"))
            if csv_files:
                try:
                    df = pd.read_csv(csv_files[0])
                    if task_code:
                        data[task_code] = df
                except Exception as e:
                    print(f"Error loading {csv_files[0]}: {e}")
        
        return data
    
    def load_all_subjects(
        self,
        rounds: Optional[List[int]] = None,
        tasks: Optional[List[str]] = None,
        max_subjects: Optional[int] = None
    ) -> List[Dict]:
        """
        Load data from all subjects.
        
        Args:
            rounds: List of rounds to load. None = all.
            tasks: List of task codes to load. None = all.
            max_subjects: Maximum number of subjects to load (for testing).
            
        Returns:
            List of dicts with subject info and data
        """
        if rounds is None:
            rounds = list(range(1, 10))
        
        all_data = []
        subject_count = 0
        
        for round_num in rounds:
            subject_ids = self.get_subject_ids(round_num)
            
            for subject_id in tqdm(subject_ids, desc=f"Round {round_num}"):
                if max_subjects and subject_count >= max_subjects:
                    return all_data
                
                for session in [1, 2]:
                    session_data = self.load_subject_session(
                        subject_id, round_num, session, tasks
                    )
                    
                    if session_data:
                        all_data.append({
                            'subject_id': subject_id,
                            'round': round_num,
                            'session': session,
                            'data': session_data
                        })
                
                subject_count += 1
        
        return all_data


def create_sequences(
    df: pd.DataFrame,
    seq_length: int = 1000,
    stride: int = 500,
    columns: List[str] = ['x', 'y', 'dP']
) -> np.ndarray:
    """
    Convert DataFrame to overlapping sequences for ML.
    
    Args:
        df: Eye tracking DataFrame
        seq_length: Length of each sequence (in samples)
        stride: Step size between sequences
        columns: Columns to include in sequences
        
    Returns:
        Array of shape (n_sequences, seq_length, n_features)
    """
    valid_mask = df['val'] == 0
    data = df[columns].values
    
    sequences = []
    for start in range(0, len(data) - seq_length + 1, stride):
        end = start + seq_length
        seq = data[start:end]
        
        valid_ratio = valid_mask.iloc[start:end].mean()
        if valid_ratio > 0.8:
            sequences.append(seq)
    
    if not sequences:
        return np.array([])
    
    return np.array(sequences)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "GazeBase_v2_0"
    
    loader = GazeBaseLoader(data_root)
    
    print("\nExtracting Round 1...")
    loader.extract_all_zips(rounds=[1])
    
    print("\nLoading sample data...")
    subjects = loader.get_subject_ids(1)
    print(f"Found {len(subjects)} subjects in Round 1")
    
    if subjects:
        sample = loader.load_subject_session(subjects[0], 1, 1, tasks=['FXS'])
        if 'FXS' in sample:
            print(f"\nSample FXS data shape: {sample['FXS'].shape}")
            print(sample['FXS'].head())
