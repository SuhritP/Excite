#!/bin/bash
# Setup script for UCSB Pod cluster
# Run this once after cloning the repo

echo "Setting up Excite on cluster..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs checkpoints

# Download GazeBase (if not already present)
if [ ! -d "GazeBase_v2_0" ]; then
    echo ""
    echo "GazeBase data not found!"
    echo "Download from: https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257"
    echo "Then extract to GazeBase_v2_0/"
    echo ""
fi

echo "Setup complete!"
echo ""
echo "To run pre-training:"
echo "  sbatch scripts/pretrain_gpu.sh"
