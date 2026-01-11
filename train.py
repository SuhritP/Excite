#!/usr/bin/env python3
"""
Main Training Script for ADHD Detection Model

Usage:
    # Pre-train on GazeBase
    python train.py pretrain --data_dir GazeBase_v2_0 --rounds 1 2 3
    
    # Fine-tune on ADHD data (when you have labeled data)
    python train.py finetune --pretrained_path checkpoints/best_pretrain.pt
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.loader import GazeBaseLoader, create_sequences
from src.data.adhd_loader import load_adhd_sequences
from src.features.extraction import FeatureExtractor, extract_adhd_features
from src.models.gaze_model import GazeEncoder, GazeClassifier, GazePretrainer
from src.training.trainer import Trainer, GazeSequenceDataset, GazeClassificationDataset


def prepare_gazebase_sequences(
    data_dir: str,
    rounds: list,
    tasks: list = ['FXS', 'RAN'],
    seq_length: int = 1000,
    stride: int = 500,
    max_subjects: int = None
) -> np.ndarray:
    """
    Load GazeBase data and convert to sequences for pre-training.
    """
    print(f"\n{'='*60}")
    print("PREPARING GAZEBASE DATA")
    print(f"{'='*60}")
    
    loader = GazeBaseLoader(data_dir)
    
    print(f"\nExtracting zips for rounds: {rounds}")
    loader.extract_all_zips(rounds=rounds)
    
    print(f"\nLoading subject data...")
    all_data = loader.load_all_subjects(
        rounds=rounds,
        tasks=tasks,
        max_subjects=max_subjects
    )
    print(f"Loaded {len(all_data)} subject-sessions")
    
    print(f"\nConverting to sequences (length={seq_length}, stride={stride})...")
    all_sequences = []
    
    for subject_data in tqdm(all_data, desc="Processing"):
        for task_code, df in subject_data['data'].items():
            sequences = create_sequences(
                df, 
                seq_length=seq_length,
                stride=stride,
                columns=['x', 'y', 'dP']
            )
            if len(sequences) > 0:
                all_sequences.append(sequences)
    
    if not all_sequences:
        raise ValueError("No valid sequences extracted!")
    
    all_sequences = np.concatenate(all_sequences, axis=0)
    print(f"\nTotal sequences: {len(all_sequences)}")
    print(f"Sequence shape: {all_sequences.shape}")
    
    return all_sequences


def pretrain(args):
    """Pre-train GazeEncoder on GazeBase."""
    print("\n" + "="*60)
    print("STAGE 1: PRE-TRAINING ON GAZEBASE")
    print("="*60)
    
    sequences = prepare_gazebase_sequences(
        data_dir=args.data_dir,
        rounds=args.rounds,
        tasks=args.tasks,
        seq_length=args.seq_length,
        stride=args.stride,
        max_subjects=args.max_subjects
    )
    
    dataset = GazeSequenceDataset(sequences, augment=True)
    
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    encoder = GazeEncoder(
        input_dim=5,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    pretrainer = GazePretrainer(encoder, output_dim=2)
    
    total_params = sum(p.numel() for p in pretrainer.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    trainer = Trainer(pretrainer, checkpoint_dir=args.checkpoint_dir)
    print(f"Device: {trainer.device}")
    
    print(f"\nStarting pre-training for {args.epochs} epochs...")
    history = trainer.pretrain(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    
    encoder_path = Path(args.checkpoint_dir) / 'encoder_pretrained.pt'
    torch.save(encoder.state_dict(), encoder_path)
    print(f"\nSaved pre-trained encoder: {encoder_path}")
    
    return history


def finetune(args):
    """Fine-tune classifier on ADHD-labeled data."""
    print("\n" + "="*60)
    print("STAGE 2: FINE-TUNING FOR ADHD CLASSIFICATION")
    print("="*60)
    
    # Load ADHD data
    adhd_path = args.adhd_data_path or "data/adhd/adhd_sequences.npz"
    print(f"\nLoading ADHD data from: {adhd_path}")
    
    try:
        sequences, labels = load_adhd_sequences(adhd_path)
    except FileNotFoundError as e:
        print(f"\n[!] {e}")
        print("\nTo extract ADHD data, run:")
        print("  python scripts/extract_adhd_data.py")
        return None
    
    # Create dataset
    dataset = GazeClassificationDataset(sequences, labels)
    
    # Split train/val
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Build model
    encoder = GazeEncoder(
        input_dim=5,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    if args.pretrained_path:
        print(f"\nLoading pre-trained encoder: {args.pretrained_path}")
        encoder.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
    
    classifier = GazeClassifier(
        encoder,
        n_classes=2,
        freeze_encoder=args.freeze_encoder
    )
    
    # Train
    trainer = Trainer(classifier, checkpoint_dir=args.checkpoint_dir)
    print(f"Device: {trainer.device}")
    
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    history = trainer.finetune(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    
    print(f"\nBest validation accuracy: {history['best_val_acc']:.3f}")
    print(f"Model saved to: {args.checkpoint_dir}/best_classifier.pt")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train ADHD detection model')
    subparsers = parser.add_subparsers(dest='command', help='Training stage')
    
    pretrain_parser = subparsers.add_parser('pretrain', help='Pre-train on GazeBase')
    pretrain_parser.add_argument('--data_dir', type=str, default='GazeBase_v2_0',
                                  help='Path to GazeBase data')
    pretrain_parser.add_argument('--rounds', type=int, nargs='+', default=[1],
                                  help='Rounds to use (e.g., 1 2 3)')
    pretrain_parser.add_argument('--tasks', type=str, nargs='+', default=['FXS', 'RAN'],
                                  help='Tasks to use (FXS, RAN, HSS, TEX, VD1, VD2, BLG)')
    pretrain_parser.add_argument('--seq_length', type=int, default=1000,
                                  help='Sequence length')
    pretrain_parser.add_argument('--stride', type=int, default=500,
                                  help='Stride between sequences')
    pretrain_parser.add_argument('--max_subjects', type=int, default=None,
                                  help='Max subjects to load (for testing)')
    pretrain_parser.add_argument('--batch_size', type=int, default=32)
    pretrain_parser.add_argument('--epochs', type=int, default=50)
    pretrain_parser.add_argument('--lr', type=float, default=1e-4)
    pretrain_parser.add_argument('--patience', type=int, default=10)
    pretrain_parser.add_argument('--d_model', type=int, default=128)
    pretrain_parser.add_argument('--n_heads', type=int, default=4)
    pretrain_parser.add_argument('--n_layers', type=int, default=4)
    pretrain_parser.add_argument('--dropout', type=float, default=0.1)
    pretrain_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    pretrain_parser.add_argument('--num_workers', type=int, default=0)
    
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tune on ADHD data')
    finetune_parser.add_argument('--pretrained_path', type=str, required=False,
                                  help='Path to pre-trained encoder')
    finetune_parser.add_argument('--adhd_data_path', type=str, required=False,
                                  help='Path to ADHD data (.mat file or labels.csv)')
    finetune_parser.add_argument('--freeze_encoder', action='store_true',
                                  help='Freeze encoder weights')
    finetune_parser.add_argument('--seq_length', type=int, default=500)
    finetune_parser.add_argument('--stride', type=int, default=250)
    finetune_parser.add_argument('--batch_size', type=int, default=32)
    finetune_parser.add_argument('--epochs', type=int, default=30)
    finetune_parser.add_argument('--lr', type=float, default=1e-4)
    finetune_parser.add_argument('--patience', type=int, default=10)
    finetune_parser.add_argument('--d_model', type=int, default=128)
    finetune_parser.add_argument('--n_heads', type=int, default=4)
    finetune_parser.add_argument('--n_layers', type=int, default=4)
    finetune_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    if args.command == 'pretrain':
        pretrain(args)
    elif args.command == 'finetune':
        finetune(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
