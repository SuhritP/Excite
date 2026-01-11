"""
ML Models for ADHD Detection from Eye-Tracking Data

Architecture:
1. GazeEncoder: Pre-trained on GazeBase to learn general gaze dynamics
2. GazeClassifier: Fine-tuned on ADHD datasets for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GazeEncoder(nn.Module):
    """
    Encoder for eye-tracking sequences.
    Uses Transformer architecture to capture temporal patterns.
    
    Pre-train this on GazeBase using self-supervised learning
    (predict next gaze position or masked positions).
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        """
        Args:
            input_dim: Number of input features (x, y, velocity, acceleration, pupil)
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Encoded representation of shape (batch, seq_len, d_model)
        """
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        return x
    
    def get_sequence_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get a single embedding vector for the entire sequence."""
        encoded = self.forward(x, mask)
        return encoded.mean(dim=1)


class GazeClassifier(nn.Module):
    """
    ADHD classifier built on top of pre-trained GazeEncoder.
    
    Fine-tune this on ADHD-labeled datasets (EEGET_RSOD, TDAH-repository).
    """
    
    def __init__(
        self,
        encoder: GazeEncoder,
        n_classes: int = 2,
        freeze_encoder: bool = False,
        hidden_dim: int = 64
    ):
        """
        Args:
            encoder: Pre-trained GazeEncoder
            n_classes: Number of output classes (2 for ADHD vs Control)
            freeze_encoder: Whether to freeze encoder weights during training
            hidden_dim: Hidden dimension for classification head
        """
        super().__init__()
        
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Logits of shape (batch, n_classes)
        """
        embedding = self.encoder.get_sequence_embedding(x, mask)
        logits = self.classifier(embedding)
        return logits
    
    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x, mask)
        return F.softmax(logits, dim=-1)


class GazePretrainer(nn.Module):
    """
    Self-supervised pre-training wrapper for GazeEncoder.
    
    Task: Predict the next gaze position given the previous positions.
    This teaches the model "normal" eye movement dynamics.
    """
    
    def __init__(self, encoder: GazeEncoder, output_dim: int = 2):
        """
        Args:
            encoder: GazeEncoder to pre-train
            output_dim: Output dimension (2 for x,y prediction)
        """
        super().__init__()
        
        self.encoder = encoder
        self.predictor = nn.Linear(encoder.d_model, output_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict next position at each timestep.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Predictions of shape (batch, seq_len, output_dim)
        """
        encoded = self.encoder(x, mask)
        predictions = self.predictor(encoded)
        return predictions


class FeatureClassifier(nn.Module):
    """
    Simple MLP classifier using extracted features (not sequences).
    Use this as a baseline or when you have limited compute.
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dims: list = [64, 32],
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    batch_size = 8
    seq_len = 1000
    input_dim = 5
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing GazeEncoder...")
    encoder = GazeEncoder(input_dim=input_dim)
    encoded = encoder(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Encoded shape: {encoded.shape}")
    
    embedding = encoder.get_sequence_embedding(x)
    print(f"  Embedding shape: {embedding.shape}")
    
    print("\nTesting GazeClassifier...")
    classifier = GazeClassifier(encoder, n_classes=2)
    logits = classifier(x)
    print(f"  Logits shape: {logits.shape}")
    
    probs = classifier.predict_proba(x)
    print(f"  Probabilities: {probs[0]}")
    
    print("\nTesting GazePretrainer...")
    pretrainer = GazePretrainer(encoder)
    predictions = pretrainer(x)
    print(f"  Predictions shape: {predictions.shape}")
    
    print("\nTesting FeatureClassifier...")
    features = torch.randn(batch_size, 22)
    feat_classifier = FeatureClassifier(input_dim=22)
    feat_logits = feat_classifier(features)
    print(f"  Feature logits shape: {feat_logits.shape}")
    
    print("\nAll tests passed!")
