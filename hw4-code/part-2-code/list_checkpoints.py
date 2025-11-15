#!/usr/bin/env python3
"""
Script to list available checkpoints for resuming training
"""
import os
import glob
import argparse

def list_checkpoints(experiment_name, model_type='ft'):
    """List available checkpoints for a given experiment"""
    checkpoint_dir = f'checkpoints/{model_type}_experiments/{experiment_name}'

    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found: {checkpoint_dir}")
        return

    print(f"Available checkpoints in {checkpoint_dir}:")
    print("-" * 50)

    # List all .pt files
    pt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    pt_files.sort()

    for pt_file in pt_files:
        filename = os.path.basename(pt_file)
        file_size = os.path.getsize(pt_file) / (1024 * 1024)  # Size in MB
        print(f"  {filename} ({file_size:.2f} MB)")

    if not pt_files:
        print("  No checkpoint files found")

    # Show available epochs for resuming
    epoch_files = [f for f in pt_files if f.startswith(os.path.join(checkpoint_dir, 'model_epoch_'))]
    if epoch_files:
        print("\nAvailable epochs to resume from:")
        for epoch_file in epoch_files:
            filename = os.path.basename(epoch_file)
            epoch_num = filename.replace('model_epoch_', '').replace('.pt', '')
            print(f"  --resume_from_epoch {epoch_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='ft_experiment',
                       help='Name of the experiment')
    parser.add_argument('--model_type', type=str, default='ft', choices=['ft', 'scr'],
                       help='Model type (ft=fine-tuned, scr=from scratch)')

    args = parser.parse_args()
    list_checkpoints(args.experiment_name, args.model_type)
