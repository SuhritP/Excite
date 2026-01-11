#!/usr/bin/env python3
"""
Quick test script to verify everything works before submitting to GPU cluster.
Run this locally first!
"""

import sys
sys.path.insert(0, '.')

def test_imports():
    print("1. Testing imports...")
    from src.data.loader import GazeBaseLoader, create_sequences
    from src.features.extraction import FeatureExtractor
    from src.models.gaze_model import GazeEncoder, GazeClassifier, GazePretrainer
    from src.training.trainer import Trainer, GazeSequenceDataset
    print("   ✓ All imports OK")

def test_model():
    print("\n2. Testing model...")
    import torch
    from src.models.gaze_model import GazeEncoder, GazePretrainer
    
    encoder = GazeEncoder(input_dim=5, d_model=64, n_heads=2, n_layers=2)
    pretrainer = GazePretrainer(encoder)
    
    x = torch.randn(4, 100, 5)
    out = pretrainer(x)
    
    assert out.shape == (4, 100, 2), f"Wrong output shape: {out.shape}"
    print(f"   ✓ Model forward pass OK (output: {out.shape})")
    
    total_params = sum(p.numel() for p in pretrainer.parameters())
    print(f"   ✓ Model has {total_params:,} parameters")

def test_features():
    print("\n3. Testing feature extraction...")
    import numpy as np
    from src.features.extraction import FeatureExtractor
    
    extractor = FeatureExtractor(sampling_rate=1000)
    x = np.cumsum(np.random.randn(1000) * 0.01)
    y = np.cumsum(np.random.randn(1000) * 0.01)
    
    vel = extractor.compute_velocity(x, y)
    bcea = extractor.compute_bcea(x, y)
    microsaccades = extractor.detect_microsaccades(x, y)
    
    print(f"   ✓ Velocity: mean={vel.mean():.2f} deg/s")
    print(f"   ✓ BCEA: {bcea:.4f} deg²")
    print(f"   ✓ Microsaccades: {len(microsaccades)} detected")

def test_data_loader():
    print("\n4. Testing data loader...")
    from src.data.loader import GazeBaseLoader
    from pathlib import Path
    
    data_path = Path("GazeBase_v2_0")
    if not data_path.exists():
        print("   ⚠ GazeBase_v2_0 not found - skipping data test")
        return
    
    loader = GazeBaseLoader(str(data_path))
    subjects = loader.get_subject_ids(1)
    print(f"   ✓ Found {len(subjects)} subjects in Round 1")
    
    if subjects:
        data = loader.load_subject_session(subjects[0], 1, 1, tasks=['FXS'])
        if 'FXS' in data:
            print(f"   ✓ Sample data shape: {data['FXS'].shape}")

def test_gpu():
    print("\n5. Testing GPU...")
    import torch
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        print("   ✓ MPS (Apple Silicon) available")
    else:
        print("   ⚠ No GPU detected - will use CPU")

def test_training_step():
    print("\n6. Testing training step...")
    import torch
    import numpy as np
    from src.models.gaze_model import GazeEncoder, GazePretrainer
    from src.training.trainer import Trainer, GazeSequenceDataset
    from torch.utils.data import DataLoader
    
    sequences = np.random.randn(20, 200, 3) * 0.1
    dataset = GazeSequenceDataset(sequences, augment=False)
    loader = DataLoader(dataset, batch_size=4)
    
    encoder = GazeEncoder(input_dim=5, d_model=32, n_heads=2, n_layers=1)
    pretrainer = GazePretrainer(encoder)
    trainer = Trainer(pretrainer, checkpoint_dir='checkpoints')
    
    print(f"   ✓ Using device: {trainer.device}")
    
    # Single training step
    trainer.model.train()
    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x.to(trainer.device)
    batch_y = batch_y.to(trainer.device)
    
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    optimizer.zero_grad()
    out = trainer.model(batch_x)
    loss = criterion(out, batch_y)
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Training step completed (loss: {loss.item():.4f})")


if __name__ == "__main__":
    print("="*50)
    print("EXCITE - Pre-flight Check")
    print("="*50)
    
    try:
        test_imports()
        test_model()
        test_features()
        test_data_loader()
        test_gpu()
        test_training_step()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED - Ready for GPU training!")
        print("="*50)
        print("\nNext steps:")
        print("  1. Upload to cluster: scp -r Excite user@pod.ucsb.edu:~/")
        print("  2. Setup: bash scripts/setup_cluster.sh")
        print("  3. Run: sbatch scripts/pretrain_gpu.sh")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
