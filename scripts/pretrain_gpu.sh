#!/bin/bash
#SBATCH --job-name=excite-pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "============================================"
echo "EXCITE - Pre-training (Fixed)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

source venv/bin/activate
mkdir -p logs checkpoints

python train.py pretrain \
    --data_dir GazeBase_v2_0 \
    --rounds 1 2 3 \
    --tasks FXS RAN \
    --seq_length 1000 \
    --stride 500 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --patience 15 \
    --checkpoint_dir checkpoints

echo "Finished: $(date)"
