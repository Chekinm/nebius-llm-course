"""N-gram language model — pure Python MLE implementation.

No external NLP libraries. Uses only Python stdlib (collections, math, random).
"""

import math
import random
from collections import Counter, defaultdict


class NgramLM:
    """Maximum Likelihood Estimator N-gram language model.

    Stores n-gram counts for all orders 1..n and computes conditional
    probabilities P(word | context) by frequency ratios.

    Attributes:
        n: The order of the model (e.g. 3 for trigrams).
        vocab: Set of all known tokens (including special tokens).
    """

    UNK = "<UNK>"
    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, n: int) -> None:
        self.n = n
        self.vocab: set[str] = set()
        # _counts[order][context_tuple][word] = frequency
        self._counts: dict[int, dict[tuple, Counter]] = {
            k: defaultdict(Counter) for k in range(1, n + 1)
        }

    def fit(self, tokenized_text: list[list[str]]) -> None:
        """Train the model on a tokenized corpus.

        Builds the vocabulary from all tokens in the corpus, then counts
        all k-grams for k from 1 to n for each sentence (with BOS/EOS padding).

        Args:
            tokenized_text: List of sentences; each sentence is a list of tokens.
        """
        all_words = [w for sent in tokenized_text for w in sent]
        self.vocab = set(all_words) | {self.UNK, self.BOS, self.EOS}

        for sent in tokenized_text:
            tokens = [w if w in self.vocab else self.UNK for w in sent]
            # Pad: (n-1) BOS tokens at the start, one EOS at the end
            padded = [self.BOS] * (self.n - 1) + tokens + [self.EOS]

            for k in range(1, self.n + 1):
                for i in range(len(padded) - k + 1):
                    kgram = tuple(padded[i : i + k])
                    context = kgram[:-1]  # empty tuple for unigrams
                    word = kgram[-1]
                    self._counts[k][context][word] += 1

    def score(self, word: str, context: list[str] | None = None) -> float:
        """Return P(word | context) using MLE.

        Trims context to n-1 words and maps OOV tokens to <UNK>.
        Returns 0.0 if the context has never been seen.

        Args:
            word: The word to score.
            context: Left context (preceding words). None means unigram.

        Returns:
            Probability as a float in [0, 1].
        """
        word = word if word in self.vocab else self.UNK
        ctx_words = (context or [])[-(self.n - 1) :]
        ctx = tuple(w if w in self.vocab else self.UNK for w in ctx_words)
        order = len(ctx) + 1

        ctx_total = sum(self._counts[order][ctx].values())
        if ctx_total == 0:
            return 0.0
        return self._counts[order][ctx][word] / ctx_total

    def logscore(self, word: str, context: list[str] | None = None) -> float:
        """Return log2 P(word | context). Returns -inf for zero probability."""
        p = self.score(word, context)
        return math.log2(p) if p > 0 else float("-inf")

    def generate(self, num_words: int, random_seed: int = 42) -> list[str]:
        """Generate a token sequence by sampling from the model.

        Starts with BOS context, samples one token at a time from the
        conditional distribution, and stops at EOS or num_words tokens.

        Args:
            num_words: Maximum number of tokens to generate.
            random_seed: Seed for reproducible results.

        Returns:
            List of generated tokens (may include EOS at the end).
        """
        rng = random.Random(random_seed)
        context = [self.BOS] * (self.n - 1)
        result = []

        for _ in range(num_words):
            ctx = tuple(context[-(self.n - 1) :])
            candidates = self._counts[self.n][ctx]
            if not candidates:
                break
            words = list(candidates.keys())
            weights = list(candidates.values())
            token = rng.choices(words, weights=weights, k=1)[0]
            result.append(token)
            if token == self.EOS:
                break
            context.append(token)

        return result


def build_model(tokenized_text: list[list[str]], n: int = 3) -> NgramLM:
    """Train an MLE n-gram language model.

    Args:
        tokenized_text: Pre-tokenized corpus as a list of sentences,
                        each sentence being a list of string tokens.
        n: The order of the n-gram model (default: 3 for trigrams).

    Returns:
        A fitted NgramLM instance.
    """
    model = NgramLM(n)
    model.fit(tokenized_text)
    return model


def score_word(
    model: NgramLM, word: str, context: list[str] | None = None
) -> float:
    """Return MLE probability P(word | context)."""
    return model.score(word, context)


def logscore_word(
    model: NgramLM, word: str, context: list[str] | None = None
) -> float:
    """Return log2 MLE probability for a word given optional context."""
    return model.logscore(word, context)
