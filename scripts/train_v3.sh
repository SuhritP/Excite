#!/bin/bash
#SBATCH --job-name=adhd-v3
#SBATCH --output=logs/train_v3_%j.out
#SBATCH --error=logs/train_v3_%j.err
#SBATCH --time=00:30:00
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
from sklearn.model_selection import StratifiedKFold

print("="*60)
print("ADHD V3 - TRANSFORMER + CROSS-VALIDATION")
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
            if np.random.random() < 0.5: s[:, :2] *= np.random.uniform(0.8, 1.2)
        x, y, p = s[:, 0], s[:, 1], s[:, 2]
        dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
        v = np.sqrt(dx**2 + dy**2) * 30
        a = np.abs(np.diff(v, prepend=v[0])) * 30
        f = np.stack([x, y, v, a, p], axis=1)
        for j in range(5):
            std = f[:, j].std()
            if std > 1e-6: f[:, j] = (f[:, j] - f[:, j].mean()) / std
        return torch.FloatTensor(np.nan_to_num(np.clip(f, -5, 5))), torch.tensor(self.labs[i])

class TransformerModel(nn.Module):
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
best_overall = 0
best_state = None

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
    print(f"\nFold {fold+1}/5")
    train_ds = DS(sequences[train_idx], labels[train_idx], aug=True)
    val_ds = DS(sequences[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    model = TransformerModel().to(device)
    w = torch.FloatTensor([1.0, len(labels[train_idx])/np.sum(labels[train_idx]==1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=w/w.sum()*2)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 60)
    
    best_fold = 0
    for ep in range(60):
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
        if acc > best_fold:
            best_fold = acc
            if acc > best_overall:
                best_overall = acc
                best_state = model.state_dict().copy()
    
    fold_accs.append(best_fold)
    print(f"  Best: {best_fold:.1%}")

print(f"\n{'='*60}")
print(f"CROSS-VAL ACCURACY: {np.mean(fold_accs):.1%} (+/- {np.std(fold_accs):.1%})")
print(f"BEST SINGLE FOLD: {best_overall:.1%}")
print(f"{'='*60}")

torch.save({'model_state_dict': best_state, 'accuracy': best_overall, 'cv_acc': np.mean(fold_accs)}, 
           'checkpoints/adhd_model.pt')
print("Saved: checkpoints/adhd_model.pt")
PYEOF
