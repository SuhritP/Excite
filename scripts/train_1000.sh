#!/bin/bash
#SBATCH --job-name=adhd-1000ep
#SBATCH --output=logs/train_1000_%j.out
#SBATCH --error=logs/train_1000_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

source venv/bin/activate
python << 'PYEOF'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

print("="*60)
print("ADHD MODEL - 1000 EPOCHS")
print("="*60)

data = np.load('data/adhd/adhd_sequences.npz')
sequences = data['sequences'].astype(np.float32)
labels = data['labels'].astype(np.int64)
print(f"Data: {len(sequences)} samples")

class DS(Dataset):
    def __init__(self, seqs, labs, aug=False):
        self.seqs, self.labs, self.aug = seqs, labs, aug
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i].copy()
        if self.aug and np.random.random() < 0.5:
            s += np.random.randn(*s.shape) * 0.1
        x, y, p = s[:, 0], s[:, 1], s[:, 2]
        dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
        v = np.sqrt(dx**2 + dy**2) * 30
        a = np.abs(np.diff(v, prepend=v[0])) * 30
        f = np.stack([x, y, v, a, p], axis=1)
        for j in range(5):
            std = f[:, j].std()
            if std > 1e-6: f[:, j] = (f[:, j] - f[:, j].mean()) / std
        return torch.FloatTensor(np.nan_to_num(np.clip(f, -5, 5))), torch.tensor(self.labs[i])

n_val = len(sequences) // 5
idx = np.random.permutation(len(sequences))
train_ds = DS(sequences[idx[n_val:]], labels[idx[n_val:]], aug=True)
val_ds = DS(sequences[idx[:n_val]], labels[idx[:n_val]])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 64)
        self.pos = nn.Parameter(torch.randn(1, 500, 64) * 0.02)
        enc = nn.TransformerEncoderLayer(64, 4, 128, 0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, 3)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))
    def forward(self, x):
        x = self.proj(x) + self.pos
        x = self.transformer(x)
        return self.head(x.mean(dim=1))

device = torch.device('cuda')
model = Model().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

w = torch.FloatTensor([1.0, len(labels)/np.sum(labels==1)]).to(device)
criterion = nn.CrossEntropyLoss(weight=w/w.sum()*2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

best_acc, patience = 0, 0
print("\nTraining 1000 epochs...")
for ep in range(1000):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    
    model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item()
            t += len(y)
    acc = c / t
    
    if ep % 50 == 0 or acc > best_acc:
        print(f"Ep {ep+1:4d}: Val Acc = {acc:.1%}", end="")
        if acc > best_acc:
            best_acc, patience = acc, 0
            torch.save({'model_state_dict': model.state_dict(), 'accuracy': acc, 'epoch': ep}, 
                      'checkpoints/adhd_model.pt')
            print(" *SAVED*")
        else:
            print()
            patience += 1
    else:
        patience += 1
    
    if patience >= 100:
        print(f"Early stopping at epoch {ep+1}")
        break

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE - BEST ACCURACY: {best_acc:.1%}")
print(f"{'='*60}")
PYEOF
