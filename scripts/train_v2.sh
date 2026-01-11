#!/bin/bash
#SBATCH --job-name=adhd-v2
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "ADHD TRAINING V2"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
source venv/bin/activate

python << 'PYEOF'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

print("="*60)
print("ADHD MODEL V2 - 100 EPOCHS")
print("="*60)

data = np.load('data/adhd/adhd_sequences.npz')
sequences = data['sequences'].astype(np.float32)
labels = data['labels'].astype(np.int64)
print(f"Data: {len(sequences)} (ADHD:{np.sum(labels==1)}, Ctrl:{np.sum(labels==0)})")

class ADHDDataset(Dataset):
    def __init__(self, sequences, labels, fps=30, augment=False):
        self.sequences, self.labels, self.fps, self.augment = sequences, labels, fps, augment
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        if self.augment and np.random.random() < 0.3:
            seq += np.random.randn(*seq.shape) * 0.05
        x, y, pupil = seq[:, 0], seq[:, 1], seq[:, 2]
        dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
        velocity = np.sqrt(dx**2 + dy**2) * self.fps
        accel = np.abs(np.diff(velocity, prepend=velocity[0])) * self.fps
        features = np.stack([x, y, velocity, accel, pupil], axis=1)
        for i in range(5):
            s = features[:, i].std()
            if s > 1e-6: features[:, i] = (features[:, i] - features[:, i].mean()) / s
        return torch.FloatTensor(np.nan_to_num(np.clip(features, -10, 10))), torch.tensor(self.labels[idx])

n_val = len(sequences) // 5
idx = np.random.permutation(len(sequences))
train_ds = ADHDDataset(sequences[idx[n_val:]], labels[idx[n_val:]], augment=True)
val_ds = ADHDDataset(sequences[idx[:n_val]], labels[idx[:n_val]])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 256, 3, batch_first=True, dropout=0.4, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
                                nn.Linear(128, 2))
    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))

device = torch.device('cuda')
model = Model().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

weights = torch.FloatTensor([len(labels)/np.sum(labels==0), len(labels)/np.sum(labels==1)]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights/weights.sum()*2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

best_acc, patience = 0, 0
print("\nTraining...")
for epoch in range(100):
    model.train()
    tc, tt = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tc += (model(x).argmax(1) == y).sum().item()
        tt += len(y)
    scheduler.step()
    
    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            vc += (model(x).argmax(1) == y).sum().item()
            vt += len(y)
    
    train_acc, val_acc = tc/tt, vc/vt
    if epoch % 10 == 0 or val_acc > best_acc:
        print(f"Ep {epoch+1:3d}: Train={train_acc:.1%} Val={val_acc:.1%}", end="")
        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            torch.save({'model_state_dict': model.state_dict(), 'accuracy': val_acc}, 'checkpoints/adhd_model.pt')
            print(" *SAVED*")
        else: print()
    else: patience += 1
    if patience >= 25: print(f"Early stop ep {epoch+1}"); break

print(f"\n{'='*60}\nBEST: {best_acc:.1%}\n{'='*60}")
PYEOF
