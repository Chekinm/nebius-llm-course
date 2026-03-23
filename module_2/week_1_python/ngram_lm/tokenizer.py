"""Tokenization utilities.

Encapsulates the try/except NLTK tokenizer setup from the notebook,
keeping it in one place so it is not repeated between corpora.
"""

import re


def _build_tokenizers():
    """Return (sent_tokenize, word_tokenize) functions.

    Tries the standard NLTK punkt tokenizer first.
    Falls back to regex sentence splitting + ToktokTokenizer.
    """
    try:
        from nltk import word_tokenize, sent_tokenize

        # Smoke-test to catch missing punkt / punkt_tab resource
        word_tokenize(sent_tokenize("This is a test sentence. Yes it is.")[0])
        return sent_tokenize, word_tokenize
    except Exception:
        from nltk.tokenize import ToktokTokenizer

        toktok = ToktokTokenizer()
        _sent_tok = lambda text: re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text)
        _word_tok = toktok.tokenize
        return _sent_tok, _word_tok


# Module-level singletons — built once on first import
_sent_tokenize, _word_tokenize = _build_tokenizers()


def tokenize_text(text: str) -> list[list[str]]:
    """Tokenize a multi-sentence document into lowercased word lists.

    Performs sentence boundary detection, then word tokenization.
    Use this for continuous prose like the Kilgarriff paper.

    Args:
        text: Raw input string (one or more sentences).

    Returns:
        List of sentences; each sentence is a list of lowercase token strings.
    """
    return [
        list(map(str.lower, _word_tokenize(sent)))
        for sent in _sent_tokenize(text)
    ]


def tokenize_sentences(sentences: list[str]) -> list[list[str]]:
    """Tokenize a pre-split list of sentences (e.g., individual tweets).

    Each string is treated as a single sentence — no sentence splitting.
    Avoids false boundaries on abbreviations like "D.C." or "U.S.A.".

    Args:
        sentences: List of raw sentence strings.

    Returns:
        List of token lists.
    """
    return [list(_word_tokenize(s)) for s in sentences]
