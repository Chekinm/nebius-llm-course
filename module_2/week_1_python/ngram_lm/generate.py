"""Text generation utilities for trained n-gram models."""

from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer

_detokenize = TreebankWordDetokenizer().detokenize


def generate_sent(
    model: MLE,
    num_words: int = 20,
    random_seed: int = 42,
) -> str:
    """Generate a human-readable sentence from an n-gram model.

    Filters out sentence padding tokens (<s>, </s>) and joins the
    remaining tokens using NLTK's TreebankWordDetokenizer.

    Args:
        model: A fitted MLE model.
        num_words: Maximum number of tokens to generate.
        random_seed: Seed for reproducible generation.

    Returns:
        Detokenized sentence string.
    """
    tokens = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == "<s>":
            continue
        if token == "</s>":
            break
        tokens.append(token)
    return _detokenize(tokens)
