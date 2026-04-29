#!/usr/bin/env python3
"""
make_tsne.py – Generate t-SNE visualisations from a saved model checkpoint.

Usage:
    python scripts/make_tsne.py \
        --model_path ./output/final_model \
        --model_id facebook/w2v-bert-2.0 \
        --output figures/tsne.png \
        --title "t-SNE of Language Embeddings"
"""
import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.data import (load_and_prepare_dataset, get_label_mappings,
                      build_preprocess_fn, encode_dataset, AudioDataCollator)
from src.model import get_input_features_key, load_feature_extractor, load_classification_model
from src.utils import extract_embeddings, plot_tsne


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="facebook/w2v-bert-2.0")
    parser.add_argument("--output", type=str, default="figures/tsne.png")
    parser.add_argument("--title", type=str, default="t-SNE of Language Embeddings")
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_key = get_input_features_key(args.model_id)
    fe = load_feature_extractor(args.model_id)

    # Load data
    _, valid_ds = load_and_prepare_dataset()
    s2i, i2s = get_label_mappings(valid_ds)
    preprocess_fn = build_preprocess_fn(fe, s2i, input_key)
    valid_enc = encode_dataset(valid_ds, preprocess_fn)

    # Load model
    model, _ = load_classification_model(args.model_id, len(s2i), s2i, i2s)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    collator = AudioDataCollator(fe, input_key)
    embeddings, lang_ids = extract_embeddings(
        model, valid_enc, collator, input_key, device, args.max_samples
    )

    plot_tsne(embeddings, lang_ids, id2label=i2s, title=args.title, save_path=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
