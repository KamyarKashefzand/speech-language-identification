#!/usr/bin/env python3
"""
train_model.py – Main training script for spoken language identification.

This script trains models to classify audio by spoken language. It supports three
training strategies via the --mode flag:

  1. standard    : Fine-tunes a pretrained audio model with a classification head
  2. dann        : Trains with Domain Adversarial Neural Networks (DANN) to learn
                   speaker-invariant language features
  3. dann_aug    : Combines DANN with speed-perturbation augmentation for robustness

Each mode uses a shared feature extractor but different training objectives. The
standard mode performs straightforward supervised learning, while DANN modes add
an auxiliary speaker classification head to encourage speaker-agnostic features.

Dependencies: transformers, torch, wandb, huggingface_hub, librosa

Usage examples:
  python train_model.py --mode standard --model_id facebook/w2v-bert-2.0 --lr 4e-5 --epochs 6
  python train_model.py --mode dann --model_id facebook/w2v-bert-2.0 --lr 4e-5 --epochs 6
  python train_model.py --mode dann_aug --model_id facebook/w2v-bert-2.0 --lr 4e-5 --epochs 6

See README.md for detailed setup instructions.
"""
import argparse
import numpy as np
import torch
import wandb

from transformers import TrainingArguments, Trainer, set_seed
from huggingface_hub import login

from src.data import (
    load_and_prepare_dataset,
    get_label_mappings,
    build_preprocess_fn,
    encode_dataset,
    AudioDataCollator,
    DANNDataCollator,
)
from src.model import get_input_features_key, load_feature_extractor, load_classification_model
from src.dann import DANNModel, DANNTrainer
from src.utils import compute_metrics

import evaluate


def parse_args():
    """Parse command-line arguments for training configuration."""
    p = argparse.ArgumentParser(description="Spoken Language Identification Training")
    
    # Training mode selection
    p.add_argument("--mode", choices=["standard", "dann", "dann_aug"], default="standard",
                   help="Training strategy: standard (baseline), dann (adversarial), dann_aug (adversarial + augmentation)")
    
    # Model and data parameters
    p.add_argument("--model_id", type=str, default="facebook/w2v-bert-2.0",
                   help="Pretrained model identifier from HuggingFace Hub")
    p.add_argument("--max_duration", type=float, default=2.0,
                   help="Maximum audio duration in seconds (longer samples are truncated)")
    
    # Training hyperparameters
    p.add_argument("--lr", type=float, default=4e-5,
                   help="Learning rate for optimizer")
    p.add_argument("--epochs", type=int, default=6,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size per device")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps")
    p.add_argument("--warmup_ratio", type=float, default=0.15,
                   help="Warmup ratio for learning rate scheduler")
    p.add_argument("--dropout", action="store_true", default=False,
                   help="Enable dropout regularization in classification head")
    
    # Logging and output
    p.add_argument("--output_dir", type=str, default="./output",
                   help="Directory to save model checkpoints and final model")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    
    # Authentication
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace Hub token for authentication (optional)")
    p.add_argument("--wandb_key", type=str, default=None,
                   help="Weights & Biases API key for experiment tracking (optional)")
    p.add_argument("--wandb_project", type=str, default="Indic-SLID",
                   help="W&B project name for organizing runs")
    
    return p.parse_args()


def main():
    """Main training pipeline orchestration."""
    args = parse_args()
    set_seed(args.seed)

    # ────────────────────────────────────────────────────────────────────
    # Environment setup
    # ────────────────────────────────────────────────────────────────────
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Authenticate with external services if credentials provided
    if args.hf_token:
        login(token=args.hf_token)
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # ────────────────────────────────────────────────────────────────────
    # Data loading and preprocessing
    # ────────────────────────────────────────────────────────────────────
    # Load feature extractor for audio preprocessing
    input_features_key = get_input_features_key(args.model_id)
    feature_extractor = load_feature_extractor(args.model_id)
    
    # Load and split dataset
    train_ds, valid_ds = load_and_prepare_dataset()
    str_to_int, int_to_str = get_label_mappings(train_ds)
    num_labels = len(str_to_int)

    # Build speaker vocabulary (only needed for DANN modes which include speaker classifier)
    speaker2id = None
    if args.mode in ("dann", "dann_aug"):
        speakers = sorted(set(train_ds["speaker_id"]))
        speaker2id = {s: i for i, s in enumerate(speakers)}
        num_speakers = len(speaker2id)
        print(f"Number of speakers: {num_speakers}")

    # Build preprocessing function with optional augmentation
    augment = args.mode == "dann_aug"
    preprocess_fn = build_preprocess_fn(
        feature_extractor, str_to_int, input_features_key,
        max_duration=args.max_duration,
        speaker2id=speaker2id,
        augment=augment,
    )

    # Encode datasets with preprocessing (keep speaker_id and language for later use)
    keep_cols = ["speaker_id", "language"]
    train_ds_enc = encode_dataset(train_ds, preprocess_fn, keep_cols)
    valid_ds_enc = encode_dataset(valid_ds, preprocess_fn, keep_cols)

    # ────────────────────────────────────────────────────────────────────
    # Model initialization (mode-specific architecture)
    # ────────────────────────────────────────────────────────────────────
    if args.mode == "standard":
        # Standard mode: single classification head on pretrained encoder
        model, config = load_classification_model(
            args.model_id, num_labels, str_to_int, int_to_str,
            apply_dropout=args.dropout,
        )
        data_collator = AudioDataCollator(feature_extractor, input_features_key)
    else:
        # DANN modes: dual-head architecture (language + speaker classifiers)
        # Both heads share the encoder to learn speaker-invariant representations
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_id)
        model = DANNModel(args.model_id, num_labels, num_speakers, config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # DANN collator batches speaker IDs alongside audio features
        data_collator = DANNDataCollator(feature_extractor, input_features_key)

    # ────────────────────────────────────────────────────────────────────
    # Training configuration and experiment tracking
    # ────────────────────────────────────────────────────────────────────
    run_name = f"{args.mode}_{args.model_id.split('/')[-1]}_lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="wandb",
        logging_steps=1,
        
        # Batch sizes and gradient accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        # Evaluation and checkpointing strategy
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,  # Keep only last 2 checkpoints
        
        # Optimization
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        
        # Hardware acceleration
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        
        # DANN batches include extra keys (speaker_ids) that must be preserved
        remove_unused_columns=False if args.mode != "standard" else True,
    )

    # ────────────────────────────────────────────────────────────────────
    # Trainer selection (mode-specific trainer with appropriate loss)
    # ────────────────────────────────────────────────────────────────────
    if args.mode == "standard":
        # Standard trainer: supervised learning only
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_enc,
            eval_dataset=valid_ds_enc,
            processing_class=feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        # DANN trainer: combines language classification loss with speaker classification loss
        trainer = DANNTrainer(
            dann_model_ref=model,
            input_features_key=input_features_key,
            args=training_args,
            train_dataset=train_ds_enc,
            eval_dataset=valid_ds_enc,
            processing_class=feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    # ────────────────────────────────────────────────────────────────────
    # Training and evaluation
    # ────────────────────────────────────────────────────────────────────
    print(f"Starting {args.mode} training...")
    trainer.train()
    
    print("Final evaluation:")
    results = trainer.evaluate()
    print(results)
    wandb.finish()

    # ────────────────────────────────────────────────────────────────────
    # Save final model artifacts
    # ────────────────────────────────────────────────────────────────────
    save_dir = f"{args.output_dir}/final_model"
    if args.mode == "standard":
        # Standard mode: use HF save_pretrained for compatibility
        model.save_pretrained(save_dir)
    else:
        # DANN mode: save raw state_dict
        torch.save(model.state_dict(), f"{save_dir}/dann_model.pt")
    
    # Feature extractor is shared across all modes
    feature_extractor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
