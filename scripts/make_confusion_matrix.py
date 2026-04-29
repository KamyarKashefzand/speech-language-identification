#!/usr/bin/env python3
"""
make_confusion_matrix.py – Generate confusion matrix from a trained model.

Usage:
    python scripts/make_confusion_matrix.py \
        --model_path ./output/final_model \
        --model_id facebook/w2v-bert-2.0 \
        --output figures/confusion_matrix.png
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.data import (load_and_prepare_dataset, get_label_mappings,
                      build_preprocess_fn, encode_dataset, AudioDataCollator)
from src.model import get_input_features_key, load_feature_extractor, load_classification_model
from transformers import Trainer, TrainingArguments
from src.utils import compute_metrics, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="facebook/w2v-bert-2.0")
    parser.add_argument("--output", type=str, default="figures/confusion_matrix.png")
    parser.add_argument("--title", type=str, default="Confusion Matrix")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_key = get_input_features_key(args.model_id)
    fe = load_feature_extractor(args.model_id)

    _, valid_ds = load_and_prepare_dataset()
    s2i, i2s = get_label_mappings(valid_ds)
    preprocess_fn = build_preprocess_fn(fe, s2i, input_key)
    valid_enc = encode_dataset(valid_ds, preprocess_fn)

    model, _ = load_classification_model(args.model_id, len(s2i), s2i, i2s)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    collator = AudioDataCollator(fe, input_key)

    # using HF Trainer.predict to handle batching and device placement for us
    training_args = TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=8)
    trainer = Trainer(model=model, args=training_args, data_collator=collator,
                      processing_class=fe, compute_metrics=compute_metrics)

    predictions = trainer.predict(valid_enc)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    label_names = [i2s[i] for i in sorted(i2s.keys())]
    plot_confusion_matrix(labels, preds, label_names, title=args.title,
                          save_path=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
