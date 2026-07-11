# -*- coding: utf-8 -*-
"""
Tiny trainable gate utilities for datasetA.

The gate intentionally uses cheap features that are already produced by the
current pipeline, so training can run on CPU without touching the ASR backbone.
"""
import json
import math

from text_norm import normalize


DEFAULT_FEATURE_NAMES = [
    "speaker_similarity",
    "intent_distance",
    "fusion_score",
    "hyp_len_norm",
    "has_hyp",
    "known_phrase",
    "very_short_hyp",
]


def make_features(speaker_similarity, hyp_text, intent_distance, fusion_weight=0.70):
    hyp = normalize(hyp_text or "")
    sim = float(speaker_similarity if speaker_similarity is not None else -1.0)
    intent = float(intent_distance if intent_distance is not None else 1.0)
    hyp_len = len(hyp)
    return {
        "speaker_similarity": sim,
        "intent_distance": intent,
        "fusion_score": sim - float(fusion_weight) * intent,
        "hyp_len_norm": min(hyp_len / 20.0, 2.0),
        "has_hyp": 1.0 if hyp_len else 0.0,
        "known_phrase": 1.0 if intent <= 0.50 else 0.0,
        "very_short_hyp": 1.0 if 0 < hyp_len <= 2 else 0.0,
    }


def vectorize(features, feature_names):
    return [float(features.get(name, 0.0)) for name in feature_names]


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict_probability(model, features):
    names = model.get("feature_names", DEFAULT_FEATURE_NAMES)
    means = model.get("feature_mean", [0.0] * len(names))
    scales = model.get("feature_scale", [1.0] * len(names))
    weights = model["weights"]
    bias = float(model.get("bias", 0.0))
    xs = vectorize(features, names)
    logit = bias
    for x, mean, scale, weight in zip(xs, means, scales, weights):
        logit += ((x - mean) / (scale or 1.0)) * weight
    return _sigmoid(logit)


def accept(model, features):
    prob = predict_probability(model, features)
    return prob >= float(model.get("threshold", 0.5)), prob


def load_gate_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_gate_model(path, model):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
