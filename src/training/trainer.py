"""
Training Pipeline for Gaze-Based ADHD Detection

Two-stage training:
1. Pre-training: Self-supervised on GazeBase (predict next gaze position)
2. Fine-tuning: Supervised on ADHD-labeled data (classification)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gaze_model import GazeEncoder, GazeClassifier, GazePretrainer
from src.features.extraction import FeatureExtractor


class GazeSequenceDataset(Dataset):
    """Dataset for gaze sequences (pre-training)."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        sampling_rate: int = 1000,
        augment: bool = True
    ):
        """
        Args:
            sequences: Array of shape (n_samples, seq_len, n_features)
                       Features: [x, y, dP] or [x, y]
            sampling_rate: Sampling rate in Hz
            augment: Whether to apply data augmentation
        """
        self.sequences = self._normalize_sequences(sequences)
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.feature_extractor = FeatureExtractor(sampling_rate)
    
    def _normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Normalize sequences to prevent NaN during training."""
        sequences = sequences.copy().astype(np.float32)
        
        # Replace inf/nan with 0
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize each feature dimension
        for i in range(sequences.shape[2]):
            feat = sequences[:, :, i]
            mean = np.mean(feat)
            std = np.std(feat)
            if std > 1e-6:
                sequences[:, :, i] = (feat - mean) / std
            else:
                sequences[:, :, i] = feat - mean
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def _compute_features(self, seq: np.ndarray) -> np.ndarray:
        """Add velocity and acceleration features to raw coordinates."""
        x, y = seq[:, 0], seq[:, 1]
        
        velocity = self.feature_extractor.compute_velocity(x, y)
        acceleration = self.feature_extractor.compute_acceleration(velocity)
        
        # Clip extreme values to prevent NaN
        velocity = np.clip(velocity, -100, 100)
        acceleration = np.clip(acceleration, -1000, 1000)
        
        if seq.shape[1] >= 3:
            pupil = seq[:, 2]
            features = np.stack([x, y, velocity, acceleration, pupil], axis=1)
        else:
            features = np.stack([x, y, velocity, acceleration], axis=1)
        
        # Final safety check
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        if not self.augment:
            return seq
        
        if np.random.random() < 0.3:
            noise = np.random.randn(*seq.shape) * 0.01
            seq = seq + noise
        
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            seq[:, :2] = seq[:, :2] * scale
        
        if np.random.random() < 0.3:
            seq[:, :2] = -seq[:, :2]
        
        return seq
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()
        seq = self._augment(seq)
        features = self._compute_features(seq)
        
        input_seq = features[:-1]
        target = features[1:, :2]
        
        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target)
        )


class GazeClassificationDataset(Dataset):
    """Dataset for ADHD classification (fine-tuning)."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        sampling_rate: int = 1000
    ):
        self.sequences = sequences
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.feature_extractor = FeatureExtractor(sampling_rate)
    
    def __len__(self):
        return len(self.sequences)
    
    def _compute_features(self, seq: np.ndarray) -> np.ndarray:
        x, y = seq[:, 0], seq[:, 1]
        velocity = self.feature_extractor.compute_velocity(x, y)
        acceleration = self.feature_extractor.compute_acceleration(velocity)
        
        if seq.shape[1] >= 3:
            pupil = seq[:, 2]
            features = np.stack([x, y, velocity, acceleration, pupil], axis=1)
        else:
            features = np.stack([x, y, velocity, acceleration], axis=1)
        
        return features
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        features = self._compute_features(seq)
        label = self.labels[idx]
        
        return (
            torch.FloatTensor(features),
            torch.LongTensor([label])[0]
        )


class Trainer:
    """Training manager for gaze models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        checkpoint_dir: str = 'checkpoints'
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10
    ) -> Dict:
        """
        Pre-train encoder using next-position prediction.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            if val_loader:
                val_loss = self._evaluate_pretrain(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint('best_pretrain.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        return self.history
    
    def _evaluate_pretrain(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 10
    ) -> Dict:
        """
        Fine-tune classifier for ADHD detection.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
            
            train_acc = correct / total
            avg_train_loss = np.mean(train_losses)
            
            if val_loader:
                val_loss, val_acc = self._evaluate_finetune(val_loader, criterion)
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}: Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save_checkpoint('best_classifier.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: Train Acc = {train_acc:.3f}")
        
        return {'best_val_acc': best_val_acc}
    
    def _evaluate_finetune(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                losses.append(loss.item())
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        return np.mean(losses), correct / total
    
    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint: {path}")


if __name__ == "__main__":
    print("Testing training pipeline...")
    
    n_samples = 100
    seq_len = 500
    sequences = np.random.randn(n_samples, seq_len, 3) * 0.1
    
    dataset = GazeSequenceDataset(sequences, augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample input shape: {x.shape}")
    print(f"Sample target shape: {y.shape}")
    
    encoder = GazeEncoder(input_dim=5)
    pretrainer = GazePretrainer(encoder)
    
    trainer = Trainer(pretrainer, checkpoint_dir='checkpoints')
    print(f"Using device: {trainer.device}")
    
    print("\nRunning 2 test epochs...")
    history = trainer.pretrain(loader, epochs=2)
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    
    print("\nAll training tests passed!")
