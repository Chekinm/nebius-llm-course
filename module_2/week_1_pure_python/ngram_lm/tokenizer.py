"""Tokenization utilities — pure Python, no external NLP libraries.

Uses regular expressions for both sentence splitting and word tokenization.
"""

import re


def sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics.

    Splits after . ! ? when followed by whitespace.

    Args:
        text: Raw input string.

    Returns:
        List of sentence strings.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(text: str) -> list[str]:
    """Tokenize a sentence into words and punctuation tokens.

    Matches words (including contractions like "don't") and treats
    punctuation characters as separate tokens.

    Args:
        text: A single sentence string.

    Returns:
        List of token strings.
    """
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)


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
        list(map(str.lower, word_tokenize(sent)))
        for sent in sent_tokenize(text)
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
    return [list(word_tokenize(s)) for s in sentences]
