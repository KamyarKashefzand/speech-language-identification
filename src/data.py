"""
data.py – Dataset loading, preprocessing, and data collation.
"""
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor


def load_and_prepare_dataset(dataset_name="badrex/nnti-dataset-full", sampling_rate=16000, seed=42):
    """Load the dataset and resample to the target sampling rate."""
    dataset = load_dataset(dataset_name)
    train_ds = dataset["train"].shuffle(seed=seed)
    valid_ds = dataset["validation"].shuffle(seed=seed)

    # the audio column needs to be cast so HF loads waveforms at 16kHz on the fly
    train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=sampling_rate))
    valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=sampling_rate))
    return train_ds, valid_ds


def get_label_mappings(train_ds):
    """Build label-to-index and index-to-label mappings."""
    labels = train_ds.unique("language")
    str_to_int = {s: i for i, s in enumerate(labels)}
    int_to_str = {i: s for s, i in str_to_int.items()}
    return str_to_int, int_to_str


def build_preprocess_fn(feature_extractor, str_to_int, input_features_key, max_duration=2.0,
                        speaker2id=None, augment=False):
    """
    Return a preprocessing function suitable for Dataset.map().

    Parameters
    ----------
    augment : bool
        If True, apply speed perturbation (0.9x / 1.1x) to 50 % of samples.
    speaker2id : dict or None
        If provided, adds speaker_id_int to each sample (needed for DANN).
    """
    def preprocess_function(examples):
        import random

        audio_arrays = []
        for x in examples["audio_filepath"]:
            audio = x["array"]

            # optionally speed-perturb half the samples to make the model
            # less reliant on speaker-specific tempo cues
            if augment:
                import librosa
                if random.random() < 0.5:
                    rate = random.choice([0.9, 1.1])
                    audio = librosa.effects.time_stretch(audio, rate=rate)
            audio_arrays.append(audio)

        # truncate all audio to max_duration seconds so batches are uniform
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            truncation=True,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            return_attention_mask=True,
        )

        # map string language labels to integer ids
        inputs["label"] = [str_to_int[x] for x in examples["language"]]
        inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
        inputs["length"] = [len(f) for f in inputs[input_features_key]]

        # speaker ids are only needed when training with DANN
        if speaker2id is not None:
            inputs["speaker_id_int"] = [speaker2id[x] for x in examples["speaker_id"]]

        return inputs

    return preprocess_function


def encode_dataset(dataset, preprocess_fn, keep_cols=None):
    """Apply the preprocessing function and remove unneeded columns."""
    if keep_cols is None:
        keep_cols = ["speaker_id", "language"]
    # process in batches of 32 for speed; drop raw columns we no longer need
    return dataset.map(
        preprocess_fn,
        remove_columns=[c for c in dataset.column_names if c not in keep_cols],
        batched=True,
        batch_size=32,
    )


# ---------------------------------------------------------------------------
# Data collators
# ---------------------------------------------------------------------------
import torch
from typing import Any, Dict, List


class AudioDataCollator:
    """Pads audio features and attention masks for standard classification."""

    def __init__(self, feature_extractor, input_features_key="input_features"):
        self.feature_extractor = feature_extractor
        self.input_features_key = input_features_key

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            self.input_features_key: [f[self.input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }
        batch = self.feature_extractor.pad(batch, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch


class DANNDataCollator(AudioDataCollator):
    """Extends AudioDataCollator with speaker IDs for DANN training."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        batch["speaker_ids"] = torch.tensor(
            [f["speaker_id_int"] for f in features], dtype=torch.long
        )
        return batch
