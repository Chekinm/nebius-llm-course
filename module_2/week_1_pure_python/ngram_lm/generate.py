"""Text generation utilities — pure Python detokenizer."""

import re

from ngram_lm.model import NgramLM


def _detokenize(tokens: list[str]) -> str:
    """Join tokens into readable text.

    Removes spaces before punctuation and fixes common contraction spacing.

    Args:
        tokens: List of word/punctuation token strings.

    Returns:
        A single readable string.
    """
    text = " ".join(tokens)
    # Remove space before sentence-ending and inline punctuation
    text = re.sub(r" ([?.!,;:'\")\]])", r"\1", text)
    # Fix contractions: "do n't" -> "don't"
    text = re.sub(r"\s'(s|t|re|ve|ll|d|m)\b", r"'\1", text)
    return text


def generate_sent(
    model: NgramLM,
    num_words: int = 20,
    random_seed: int = 42,
) -> str:
    """Generate a human-readable sentence from an n-gram model.

    Filters out sentence padding tokens (<s>, </s>) and joins the
    remaining tokens into a readable string.

    Args:
        model: A fitted NgramLM model.
        num_words: Maximum number of tokens to generate.
        random_seed: Seed for reproducible generation.

    Returns:
        Detokenized sentence string.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == "<s>":
            continue
        if token == "</s>":
            break
        content.append(token)
    return _detokenize(content)
