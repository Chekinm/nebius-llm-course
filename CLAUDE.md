# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Nebius LLM Course** repository. Currently contains Module 2, Week 1: N-gram Language Models with NLTK.

## Development Setup

This project uses [`uv`](https://docs.astral.sh/uv/) as the package manager.

```bash
# Install dependencies
cd module_2/week_1
uv sync

# Run main entry point
uv run python main.py

# Launch Jupyter for notebook work
uv run jupyter notebook
```

Requires Python 3.14+.

## Repository Structure

```
module_2/
  week_1/           # N-gram Language Models with NLTK
    Ngram_Language_Model_with_NLTK.ipynb  # Main educational notebook
    main.py                               # Placeholder entry point
    kilgariff_ngram_model.pkl             # Pre-trained MLE model (dill-serialized)
    language-never-random.txt             # Training corpus (Kilgarriff paper)
    Donald-Tweets!.csv                    # Trump tweet dataset
    pyproject.toml                        # Project config
    uv.lock                               # Locked dependencies
```

## Architecture Notes

### Notebook Structure (`Ngram_Language_Model_with_NLTK.ipynb`)

The notebook is self-contained and educational. Flow:
1. **Preprocessing** — NLTK padding utilities, `padded_everygram_pipeline`
2. **Training** — MLE model on Kilgarriff academic paper corpus (vocabulary: 1419 items, 18741 ngrams)
3. **Scoring** — `model.score()`, `model.logscore()`, conditional probability P(word|context)
4. **Generation** — Custom `generate_sent()` filtering out padding tokens
5. **Serialization** — Uses `dill` (not standard `pickle`) to save/load models; pre-built model is `kilgariff_ngram_model.pkl`
6. **Second corpus** — Repeats training/generation on Trump tweet dataset

### Key Technical Details

- **Use `dill` not `pickle`** for model serialization — NLTK MLE models contain lambdas that pickle cannot handle
- Unknown tokens are handled via `"<UNK>"` — models trained with `padded_everygram_pipeline` automatically handle OOV words
- Sentence boundaries use `"<s>"` and `"</s>"` padding tokens; filter these out when generating human-readable text
