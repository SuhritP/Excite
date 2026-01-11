#!/bin/bash
# Quick GPU test - run this first to verify everything works
# Usage: bash scripts/quick_test.sh

echo "Quick GPU Test"
echo "=============="

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run minimal pre-training (2 epochs, 5 subjects)
python train.py pretrain \
    --data_dir GazeBase_v2_0 \
    --rounds 1 \
    --max_subjects 5 \
    --batch_size 8 \
    --epochs 2 \
    --num_workers 0

echo "Test complete!"
