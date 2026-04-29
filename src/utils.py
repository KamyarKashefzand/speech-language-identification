"""
utils.py – Shared utilities for evaluation, embedding extraction, and visualisation.
"""
import numpy as np
import torch
import evaluate
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

_accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute accuracy for an EvalPrediction tuple."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return _accuracy_metric.compute(predictions=predictions,
                                    references=eval_pred.label_ids)


# ---------------------------------------------------------------------------
# Embedding extraction (for t-SNE and speaker probes)
# ---------------------------------------------------------------------------

def extract_embeddings(model, dataset, data_collator, input_features_key,
                       device="cuda", max_samples=2000, batch_size=8):
    """
    Extract mean-pooled embeddings from the encoder.

    Works with both standard HF models and DANNModel instances.
    Returns (embeddings, lang_ids) as numpy arrays.
    """
    model.eval()
    all_emb, all_lang = [], []
    n = min(max_samples, len(dataset))
    subset = dataset.select(range(n))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_items = [subset[i] for i in range(start, end)]
        batch = data_collator(batch_items)

        with torch.no_grad():
            # our DANN model wraps the encoder directly, so we can grab it
            if hasattr(model, "encoder"):
                outputs = model.encoder(
                    input_features=batch[input_features_key].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                hidden = outputs.last_hidden_state.mean(dim=1)
            else:
                # for a standard HF classification model we need to dig into
                # the base model to get the raw hidden states (not the logits)
                base = getattr(model, model.config.model_type.replace("-", "_"), model)
                outputs = base(
                    **{input_features_key: batch[input_features_key].to(device),
                       "attention_mask": batch["attention_mask"].to(device)}
                )
                hidden = outputs.last_hidden_state.mean(dim=1)

        all_emb.append(hidden.cpu().numpy())
        all_lang.extend(batch["labels"].numpy().tolist())

    return np.concatenate(all_emb, axis=0), np.array(all_lang)


# ---------------------------------------------------------------------------
# t-SNE plotting
# ---------------------------------------------------------------------------

def plot_tsne(embeddings, labels, id2label=None, title="t-SNE", save_path=None):
    """Run t-SNE on embeddings and produce a scatter plot."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique))

    for idx, lang_id in enumerate(unique):
        mask = labels == lang_id
        lbl = id2label[lang_id] if id2label else str(lang_id)
        plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(idx)],
                    label=lbl, s=12, alpha=0.6)

    plt.title(title)
    plt.legend(fontsize=6, ncol=3, loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved t-SNE plot to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, label_names, title="Confusion Matrix",
                          save_path=None):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()
