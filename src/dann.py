"""
dann.py – Domain Adversarial Neural Network (DANN) components.

Implements the Gradient Reversal Layer and dual-head DANN model for
speaker-invariant language identification, following Ganin et al. (2016).
"""
import math
import torch
import torch.nn as nn
from transformers import AutoConfig, Trainer
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertModel


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Reverses the gradient during the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()  # forward pass is just identity

    @staticmethod
    def backward(ctx, grad_output):
        # flip the gradient so the encoder learns to *confuse* the speaker head
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# DANN lambda schedule
# ---------------------------------------------------------------------------

def dann_lambda_schedule(current_step: int, total_steps: int) -> float:
    """
    Standard DANN schedule from Ganin et al. (2016): λ(p) = 2 / (1 + exp(-10p)) - 1.
    Starts near 0 and ramps up to 1, so the adversarial signal kicks in gradually.
    """
    p = current_step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ---------------------------------------------------------------------------
# DANN Model
# ---------------------------------------------------------------------------

class DANNModel(nn.Module):
    """
    Dual-head DANN for speaker-invariant language identification.

    Architecture
    ------------
    - Shared encoder      : Wav2Vec2BertModel (fine-tuned)
    - Language head        : Linear → Softmax
    - Speaker head (GRL)  : GRL → Linear → ReLU → Dropout → Linear → Softmax
    """

    def __init__(self, model_id: str, num_labels: int, num_speakers: int,
                 config=None):
        super().__init__()
        if config is not None:
            self.encoder = Wav2Vec2BertModel.from_pretrained(model_id, config=config)
        else:
            self.encoder = Wav2Vec2BertModel.from_pretrained(model_id)

        hidden_dim = self.encoder.config.hidden_size

        # Language classifier
        self.lang_projector = nn.Linear(hidden_dim, hidden_dim)
        self.lang_classifier = nn.Linear(hidden_dim, num_labels)

        # Speaker classifier with GRL
        self.grl = GradientReversalLayer()
        self.speaker_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_speakers),
        )

    def forward(self, input_features, attention_mask=None):
        outputs = self.encoder(input_features=input_features,
                               attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, D)
        pooled = hidden.mean(dim=1)          # mean-pool over time → (B, D)

        # language branch – straightforward classification
        lang_proj = torch.relu(self.lang_projector(pooled))
        lang_logits = self.lang_classifier(lang_proj)

        # speaker branch – GRL flips gradients so the encoder *unlearns* speaker info
        reversed_pooled = self.grl(pooled)
        speaker_logits = self.speaker_head(reversed_pooled)

        return {"lang_logits": lang_logits, "speaker_logits": speaker_logits}


# ---------------------------------------------------------------------------
# DANN Trainer
# ---------------------------------------------------------------------------

class DANNTrainer(Trainer):
    """
    Hugging Face Trainer subclass for DANN training.

    Loss: L_total = L_lang + λ(t) · L_spk
    """

    def __init__(self, dann_model_ref, input_features_key="input_features",
                 *args, **kwargs):
        super().__init__(model=dann_model_ref, *args, **kwargs)
        self.dann_model_ref = dann_model_ref
        self.input_features_key = input_features_key
        self.lang_criterion = nn.CrossEntropyLoss()
        self.spk_criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_feats = inputs[self.input_features_key]
        attention_mask = inputs.get("attention_mask", None)
        lang_labels = inputs["labels"]
        speaker_ids = inputs["speaker_ids"]

        # update lambda according to the schedule – it ramps from ~0 to ~1
        total_steps = self.state.max_steps if self.state.max_steps > 0 else 1
        lambda_val = dann_lambda_schedule(self.state.global_step, total_steps)
        model.grl.set_lambda(lambda_val)

        result = model(input_features=input_feats, attention_mask=attention_mask)
        lang_logits = result["lang_logits"]
        speaker_logits = result["speaker_logits"]

        loss_lang = self.lang_criterion(lang_logits, lang_labels)
        loss_spk = self.spk_criterion(speaker_logits, speaker_ids)

        # combined loss: language task + weighted adversarial speaker task
        total_loss = loss_lang + lambda_val * loss_spk

        if self.state.global_step % 10 == 0:
            lang_acc = (lang_logits.argmax(dim=-1) == lang_labels).float().mean().item()
            spk_acc = (speaker_logits.argmax(dim=-1) == speaker_ids).float().mean().item()
            self.log({
                "dann_lambda": lambda_val,
                "loss_lang": loss_lang.item(),
                "loss_spk": loss_spk.item(),
                "train_lang_acc": lang_acc,
                "train_spk_acc": spk_acc,
            })

        if return_outputs:
            return total_loss, lang_logits
        return total_loss
