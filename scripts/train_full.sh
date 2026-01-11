#!/bin/bash
#SBATCH --job-name=adhd-full
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "ADHD FULL TRAINING - ALL DATA"
echo "Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name --format=csv,noheader

source venv/bin/activate

python << 'PYEOF'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

print("="*60)
print("ADHD MODEL TRAINING - FULL PIPELINE")
print("="*60)

# ============================================================
# STEP 1: LOAD ALL ADHD DATA
# ============================================================
print("\n[1] Loading ADHD data...")
data = np.load('data/adhd/adhd_sequences.npz')
sequences = data['sequences'].astype(np.float32)  # (N, 500, 3) = [x, y, pupil]
labels = data['labels'].astype(np.int64)

print(f"    Total samples: {len(sequences)}")
print(f"    ADHD: {np.sum(labels==1)}, Control: {np.sum(labels==0)}")
print(f"    Sequence shape: {sequences.shape}")

# ============================================================
# STEP 2: CREATE DATASET (5 features: x, y, velocity, accel, pupil)
# ============================================================
print("\n[2] Creating dataset with 5 features...")

class ADHDDataset(Dataset):
    def __init__(self, sequences, labels, fps=30):
        self.sequences = sequences
        self.labels = labels
        self.fps = fps
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (500, 3) = [x, y, pupil]
        label = self.labels[idx]
        
        x, y, pupil = seq[:, 0], seq[:, 1], seq[:, 2]
        
        # Compute velocity (degrees per frame -> degrees per second)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        velocity = np.sqrt(dx**2 + dy**2) * self.fps
        
        # Compute acceleration
        acceleration = np.abs(np.diff(velocity, prepend=velocity[0])) * self.fps
        
        # Stack: [x, y, velocity, acceleration, pupil]
        features = np.stack([x, y, velocity, acceleration, pupil], axis=1)
        
        # Normalize each feature
        for i in range(5):
            std = features[:, i].std()
            if std > 1e-6:
                features[:, i] = (features[:, i] - features[:, i].mean()) / std
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.FloatTensor(features), torch.LongTensor([label])[0]

dataset = ADHDDataset(sequences, labels, fps=30)

# Split 80/20
n_val = len(dataset) // 5
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], 
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")

# ============================================================
# STEP 3: BUILD MODEL
# ============================================================
print("\n[3] Building model...")

class ADHDClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        
        # LSTM to capture temporal patterns
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 5)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(hidden)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ADHDClassifier().to(device)
print(f"    Device: {device}")
print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# STEP 4: TRAIN
# ============================================================
print("\n[4] Training...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_acc = 0
patience = 0
max_patience = 10

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += len(y)
    
    train_acc = train_correct / train_total
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item()
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += len(y)
    
    val_acc = val_correct / val_total
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1:2d}: Train Acc={train_acc:.1%}, Val Acc={val_acc:.1%}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': val_acc,
            'epoch': epoch
        }, 'checkpoints/adhd_model.pt')
        print(f"         -> Saved (best={best_acc:.1%})")
    else:
        patience += 1
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("\n" + "="*60)
print(f"TRAINING COMPLETE - Best Accuracy: {best_acc:.1%}")
print("Model saved: checkpoints/adhd_model.pt")
print("="*60)
PYEOF
