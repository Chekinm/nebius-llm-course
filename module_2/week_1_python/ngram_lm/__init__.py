"""N-gram Language Model package — NLTK-based MLE trigram model."""

from ngram_lm.model import build_model
from ngram_lm.generate import generate_sent
from ngram_lm.serialize import save_model, load_model

__all__ = ["build_model", "generate_sent", "save_model", "load_model"]
