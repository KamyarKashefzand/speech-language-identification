"""
augmentation.py – Data augmentation utilities for bias mitigation.

Implements speed perturbation using librosa, applied probabilistically
during preprocessing.
"""
import random
import numpy as np


def speed_perturb(audio: np.ndarray, prob: float = 0.5,
                  rates=(0.9, 1.1)) -> np.ndarray:
    """
    Apply random speed perturbation to an audio waveform.

    Parameters
    ----------
    audio : np.ndarray
        Raw waveform (1-D).
    prob : float
        Probability of applying perturbation.
    rates : tuple
        Pool of stretch/compress rates to sample from.

    Returns
    -------
    np.ndarray
        Possibly time-stretched waveform.
    """
    if random.random() < prob:
        import librosa
        rate = random.choice(list(rates))
        audio = librosa.effects.time_stretch(audio, rate=rate)
    return audio
