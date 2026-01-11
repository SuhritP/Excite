#!/bin/bash
#SBATCH --job-name=adhd-finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

echo "ADHD Fine-tuning (10% data, GPU)"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

source venv/bin/activate

python << 'PYEOF'
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset

from src.data.adhd_loader import load_adhd_sequences
from src.models.gaze_model import GazeEncoder, GazeClassifier
from src.training.trainer import GazeClassificationDataset, Trainer

print("="*50)
print("ADHD FINE-TUNING (10% DATA - HACKATHON MODE)")
print("="*50)

# Load data
sequences, labels = load_adhd_sequences('data/adhd/adhd_sequences.npz')

# USE ONLY 10% OF DATA
n_samples = len(sequences) // 10
idx = np.random.permutation(len(sequences))[:n_samples]
sequences = sequences[idx]
labels = labels[idx]
print(f"\nUsing 10%: {len(sequences)} samples")
print(f"ADHD: {np.sum(labels==1)}, Control: {np.sum(labels==0)}")

# Dataset
dataset = GazeClassificationDataset(sequences, labels)
n_val = max(1, len(dataset) // 5)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Model
encoder = GazeEncoder(input_dim=5, d_model=128, n_heads=4, n_layers=4)
encoder.load_state_dict(torch.load('checkpoints/encoder_pretrained.pt', map_location='cpu'))
classifier = GazeClassifier(encoder, n_classes=2, freeze_encoder=False)

# Train on GPU
trainer = Trainer(classifier, checkpoint_dir='checkpoints')
print(f"Device: {trainer.device}")

print("\nTraining 10 epochs...")
history = trainer.finetune(train_loader, val_loader, epochs=10, lr=1e-4, patience=5)

print("\n" + "="*50)
print(f"BEST ACCURACY: {history['best_val_acc']*100:.1f}%")
print("="*50)
print("Model saved: checkpoints/best_classifier.pt")
PYEOF

echo "Done: $(date)"
