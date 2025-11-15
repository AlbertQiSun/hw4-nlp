#!/bin/bash
set -e

# Part 1: BERT Fine-tuning (Now works with CUDA fallback to CPU)
conda activate hw4-part-1-nlp
cd hw4-code/part-1-code

# Q1: Train BERT on original IMDB and evaluate on original test set
python main.py --train --eval --learning_rate 5e-5 --num_epochs 3 --batch_size 16

# Q2: Evaluate on transformed test set (should drop accuracy >4%)
python main.py --eval_transformed --model_dir ./out

# Q3: Train on augmented data (original + 5k transformed examples)
python main.py --train_augmented --learning_rate 5e-5 --num_epochs 3 --batch_size 16
python main.py --eval --model_dir ./out_augmented
python main.py --eval_transformed --model_dir ./out_augmented

# Part 2: T5 Fine-tuning (Also has CUDA fallback)
conda activate hw4-part-2-nlp
cd ../part-2-code
# T5 Training with automatic checkpointing after each epoch
python train_t5.py --finetune --learning_rate 1e-4 --max_n_epochs 10 --batch_size 8 --test_batch_size 8 --patience_epochs 3 --experiment_name ft_experiment --pretrained_model_path /gpfsnyu/scratch/qs2196/t5-small-cache --local_files_only

# To resume training from a specific epoch (e.g., epoch 5):
# python train_t5.py --finetune --learning_rate 1e-4 --max_n_epochs 10 --batch_size 8 --test_batch_size 8 --patience_epochs 3 --experiment_name ft_experiment --pretrained_model_path /gpfsnyu/scratch/qs2196/t5-small-cache --local_files_only --resume_from_epoch 5