"""N-gram language model training.

Wraps the NLTK MLE model with a clean interface.
All preprocessing details (padded everygram pipeline) are handled here.
"""

from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline


def build_model(tokenized_text: list[list[str]], n: int = 3) -> MLE:
    """Train an MLE n-gram language model.

    Args:
        tokenized_text: Pre-tokenized corpus as a list of sentences,
                        each sentence being a list of string tokens.
        n: The order of the n-gram model (default: 3 for trigrams).

    Returns:
        A fitted nltk.lm.MLE model instance.
    """
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_sents)
    return model


def score_word(
    model: MLE, word: str, context: list[str] | None = None
) -> float:
    """Return MLE probability P(word | context).

    Args:
        model: A fitted MLE model.
        word: The word to score.
        context: Optional list of preceding words (left context).
                 If None, returns the unigram probability.

    Returns:
        Probability score as float.
    """
    return model.score(word, context)


def logscore_word(
    model: MLE, word: str, context: list[str] | None = None
) -> float:
    """Return log2 MLE probability for a word given optional context."""
    return model.logscore(word, context)
