"""Shared text and label preparation helpers for ToxiGuard.

Keeping this logic in one small module is important: the dashboard must clean
new comments in exactly the same way that the training script cleaned comments.
"""

from __future__ import annotations

import re

import pandas as pd


LABEL_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def clean_text(value: object) -> str:
    """Return the lightweight text representation used by the TF-IDF model.

    The function deliberately does not remove common words or negation. In
    short comments, words such as "not" can change meaning and are useful
    signal for a traditional text classifier.
    """
    if not isinstance(value, str):
        return ""

    text = value.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[\[.*?\]\]", " ", text)
    text = re.sub(r"[^a-z0-9'!?\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_binary_label(frame: pd.DataFrame) -> pd.Series:
    """Return a binary label using the available Jigsaw toxicity columns."""
    available = [column for column in LABEL_COLUMNS if column in frame.columns]
    if not available:
        raise ValueError(
            "No supported toxicity label was found. Expected at least one of: "
            + ", ".join(LABEL_COLUMNS)
        )
    labels = frame[available].fillna(0).astype(float)
    return (labels.max(axis=1) > 0).astype(int)
