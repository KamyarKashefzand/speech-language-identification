#!/usr/bin/env python3
"""
make_plots.py – Generate training curve plots from HuggingFace trainer_state JSON logs.

Reads all *_trainer_state_*_run*.json files from a results directory, extracts
training loss, validation accuracy, and validation loss curves, and saves
publication-quality PNG plots to an output directory.

Usage:
    python scripts/make_plots.py --results_dir ../results --output_dir ../report/figures
"""
import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ── Plot styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = plt.cm.tab10.colors


def load_trainer_state(path):
    """Load a trainer_state JSON and return log_history + metadata."""
    with open(path) as f:
        data = json.load(f)
    return data


def smooth(values, weight=0.9):
    """Exponential moving average smoothing — makes noisy training loss readable."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def extract_curves(log_history):
    """Extract train loss, eval accuracy, and eval loss from log_history."""
    train_steps, train_loss = [], []
    eval_steps, eval_acc, eval_loss = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_accuracy" in entry:
            eval_steps.append(entry["step"])
            eval_acc.append(entry["eval_accuracy"])
            eval_loss.append(entry["eval_loss"])

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_acc": eval_acc,
        "eval_loss": eval_loss,
    }


def parse_run_name(filename):
    """Parse a filename into a human-readable run label."""
    base = os.path.basename(filename).replace(".json", "")
    # Extract model name and run number
    if "mms_300m" in base:
        model = "MMS-300M"
    elif "mHuBERT" in base:
        model = "mHuBERT-147"
    elif "wav2vec2-xls-r" in base:
        model = "XLS-R-300M"
    elif "w2v_bert" in base:
        model = "W2V-BERT-2.0"
    else:
        model = base.split("_trainer")[0]

    # Extract run number
    run = "run01"
    for part in base.split("_"):
        if part.startswith("run"):
            run = part
            break

    return f"{model} ({run})"


def plot_single(ax, steps, values, label, color, ylabel, title, smooth_it=False):
    """Plot a single curve on given axes."""
    if smooth_it and len(values) > 5:
        values = smooth(values)
    ax.plot(steps, values, label=label, color=color, linewidth=1.5)


def make_comparative_plot(all_curves, ylabel, title, out_path, smooth_it=False):
    """Create a single plot comparing multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (label, curves) in enumerate(all_curves.items()):
        color = COLORS[idx % len(COLORS)]
        if "train" in ylabel.lower():
            steps, vals = curves["train_steps"], curves["train_loss"]
        elif "accuracy" in ylabel.lower():
            steps, vals = curves["eval_steps"], curves["eval_acc"]
        else:
            steps, vals = curves["eval_steps"], curves["eval_loss"]

        if smooth_it and len(vals) > 5:
            vals = smooth(vals, weight=0.85)

        ax.plot(steps, vals, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_individual_plot(label, curves, out_dir):
    """Generate per-run plots: smoothed training loss, val acc, val loss."""
    safe_name = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")

    # Smoothed training loss
    if curves["train_steps"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curves["train_steps"], smooth(curves["train_loss"], 0.9),
                linewidth=1.5, color=COLORS[0])
        ax.set_xlabel("Steps")
        ax.set_ylabel("Training Loss (smoothed)")
        ax.set_title(f"{label} – Smoothed Training Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_Smoothed_Training_Loss_vs_Steps.png")
        plt.savefig(path, dpi=200)
        plt.close()

    # Validation accuracy
    if curves["eval_steps"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curves["eval_steps"], curves["eval_acc"], "o-",
                linewidth=1.5, color=COLORS[1])
        ax.set_xlabel("Steps")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"{label} – Validation Accuracy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_Validation_Accuracy_vs_Steps.png")
        plt.savefig(path, dpi=200)
        plt.close()

    # Validation loss
    if curves["eval_steps"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curves["eval_steps"], curves["eval_loss"], "o-",
                linewidth=1.5, color=COLORS[2])
        ax.set_xlabel("Steps")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"{label} – Validation Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_Validation_Loss_vs_Steps.png")
        plt.savefig(path, dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing trainer_state JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plot PNGs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover JSON files
    patterns = [
        os.path.join(args.results_dir, "*_trainer_state_*_run*.json"),
        os.path.join(args.results_dir, "trainer_state_*_run*.json"),
    ]
    json_files = []
    for pat in patterns:
        json_files.extend(glob.glob(pat))
    json_files = sorted(set(json_files))
    print(f"Found {len(json_files)} trainer state files")

    # ── Pick representative files (highest checkpoint per run) ────
    # Group by run identifier
    run_groups = defaultdict(list)
    for f in json_files:
        base = os.path.basename(f)
        # Extract run id (e.g., run01, run02)
        run_id = None
        for part in base.replace(".json", "").split("_"):
            if part.startswith("run"):
                run_id = part
        # Extract model prefix
        model_prefix = base.split("_trainer_state")[0]
        key = f"{model_prefix}_{run_id}" if run_id else model_prefix
        run_groups[key].append(f)

    # when we have multiple checkpoints for the same run, just pick the latest one
    selected = {}
    for key, files in run_groups.items():
        def ckpt_num(fp):
            base = os.path.basename(fp).replace(".json", "")
            parts = base.split("_")
            for p in parts:
                if p.isdigit():
                    return int(p)
            return 0
        best_file = max(files, key=ckpt_num)
        data = load_trainer_state(best_file)
        label = parse_run_name(best_file)
        selected[label] = extract_curves(data["log_history"])

    print(f"Selected {len(selected)} runs for plotting: {list(selected.keys())}")

    # ── Individual plots ─────────────────────────────────────────
    print("\nGenerating individual plots...")
    for label, curves in selected.items():
        make_individual_plot(label, curves, args.output_dir)

    # ── Comparative plots ────────────────────────────────────────
    print("\nGenerating comparative plots...")

    # 1. Run01 – all 4 models
    run01 = {k: v for k, v in selected.items() if "run01" in k}
    if run01:
        make_comparative_plot(
            run01, "Validation Accuracy",
            "Run01 – Model Comparison (Validation Accuracy)",
            os.path.join(args.output_dir, "model_comparison_accuracy.png"),
        )
        make_comparative_plot(
            run01, "Validation Loss",
            "Run01 – Model Comparison (Validation Loss)",
            os.path.join(args.output_dir, "model_comparison_loss.png"),
        )

    # 2. W2V-BERT runs comparison
    bert_runs = {k: v for k, v in selected.items() if "W2V-BERT" in k}
    if bert_runs:
        make_comparative_plot(
            bert_runs, "Validation Accuracy",
            "W2V-BERT-2.0 – Tuning Runs Comparison",
            os.path.join(args.output_dir, "w2v_bert_runs_accuracy_comparison.png"),
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
