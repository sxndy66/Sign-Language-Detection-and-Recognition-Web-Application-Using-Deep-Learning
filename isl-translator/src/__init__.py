"""ISL ⇄ TEXT — Indian Sign Language recognition pipeline.

A temporal transformer reads MediaPipe landmark sequences and a gated state
machine commits words into sentences. Every metric in this package is measured
on a held-out set; nothing is invented.

Public surface:
    src.model          — ISLTransformer / ISLBiLSTM sequence models
    src.preprocessing  — landmark normalization + 186-D feature extraction
    src.inference      — the confidence-filtered, EMA-smoothed state machine
"""

__version__ = "0.1.0"
