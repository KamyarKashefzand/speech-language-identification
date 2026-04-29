"""
model.py – Model initialisation helpers.
"""
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor


def get_input_features_key(model_id: str) -> str:
    """Return the correct input key for the given model."""
    # w2v-bert-2.0 expects 'input_features', all other wav2vec2 models use 'input_values'
    if model_id == "facebook/w2v-bert-2.0":
        return "input_features"
    return "input_values"


def load_feature_extractor(model_id: str):
    """Load and return the feature extractor for the given model."""
    return AutoFeatureExtractor.from_pretrained(
        model_id,
        do_normalize=True,
        return_attention_mask=True,
    )


def load_classification_model(model_id: str, num_labels: int,
                               label2id: dict, id2label: dict,
                               apply_dropout: bool = False):
    """
    Load a pretrained audio classification model and configure the
    classification head.

    Parameters
    ----------
    apply_dropout : bool
        If True, set dropout probabilities to 0.1 across all dropout types.
    """
    config = AutoConfig.from_pretrained(model_id)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    # set all dropout types at once — we found 0.1 too aggressive for this small dataset
    if apply_dropout:
        config.hidden_dropout = 0.1
        config.attention_dropout = 0.1
        config.activation_dropout = 0.1
        config.feat_proj_dropout = 0.1

    model = AutoModelForAudioClassification.from_pretrained(model_id, config=config)
    return model, config
