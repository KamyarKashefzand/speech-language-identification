#!/usr/bin/env python3
"""
plot_model_comparison.py – Compare validation accuracy across all four models
evaluated in run01.

Reads trainer_state JSON files for each model and plots accuracy curves
on a single figure.

Usage:
    python scripts/plot_model_comparison.py --results_dir ../results --output figures/model_comparison_accuracy.png
"""
import argparse
import json
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# these are the exact json filenames from run01 — one per model
MODEL_FILES = {
    "MMS-300M": "mms_300m_trainer_state_1632_run01.json",
    "mHuBERT-147": "mHuBERT-147_trainer_state_1632_run01.json",
    "XLS-R-300M": "wav2vec2-xls-r-300m_trainer_state_1632_run01.json",
    "W2V-BERT-2.0": "w2v_bert_trainer_state_1632_run01.json",
}

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/model_comparison_accuracy.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (model_name, filename) in enumerate(MODEL_FILES.items()):
        path = os.path.join(args.results_dir, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {model_name}")
            continue

        with open(path) as f:
            data = json.load(f)

        logs = data.get("log_history", [])
        steps = [e["step"] for e in logs if "eval_accuracy" in e]
        accs = [e["eval_accuracy"] for e in logs if "eval_accuracy" in e]

        ax.plot(steps, accs, "o-", label=model_name, color=COLORS[idx],
                linewidth=2, markersize=4)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Run01 – Model Comparison: Validation Accuracy")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
