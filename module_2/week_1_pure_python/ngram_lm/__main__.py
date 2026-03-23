"""CLI entry point for the ngram_lm package.

Usage:
    uv run python -m ngram_lm <command> [options]
    uv run ngram-lm <command> [options]

Commands:
    train       Train a model on a corpus and save it to disk
    generate    Load a saved model and generate text
    score       Load a saved model and score a word / context
    demo        Run the full pipeline end-to-end (train + generate both corpora)
"""

import argparse
import pathlib

_PROJECT_DIR = pathlib.Path(__file__).parent.parent  # .../week_1_pure_python/
_MODELS_DIR = _PROJECT_DIR / "models"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    from ngram_lm.data import load_kilgarriff, load_trump_tweets
    from ngram_lm.tokenizer import tokenize_text, tokenize_sentences
    from ngram_lm.model import build_model
    from ngram_lm.serialize import save_model

    print(f"Loading corpus: {args.corpus}")
    if args.corpus == "kilgarriff":
        raw = load_kilgarriff()
        tokens = tokenize_text(raw)
        default_out = _MODELS_DIR / "kilgarriff_ngram_model.pkl"
    else:  # trump
        tweets = load_trump_tweets()
        tokens = tokenize_sentences(tweets)
        default_out = _MODELS_DIR / "trump_ngram_model.pkl"

    out_path = pathlib.Path(args.output) if args.output else default_out
    print(f"Training {args.n}-gram MLE model on {len(tokens)} sentences...")
    model = build_model(tokens, n=args.n)
    print(f"Vocabulary size: {len(model.vocab)}")
    save_model(model, out_path)


def cmd_generate(args: argparse.Namespace) -> None:
    from ngram_lm.serialize import load_model
    from ngram_lm.generate import generate_sent

    model = load_model(args.model)
    for seed in range(args.seed, args.seed + args.count):
        text = generate_sent(model, num_words=args.words, random_seed=seed)
        print(f"[seed={seed}] {text}")


def cmd_score(args: argparse.Namespace) -> None:
    from ngram_lm.serialize import load_model
    from ngram_lm.model import score_word, logscore_word

    model = load_model(args.model)
    context = args.context or None
    prob = score_word(model, args.word, context)
    log_prob = logscore_word(model, args.word, context)
    ctx_str = f"| {' '.join(context)}" if context else "(unigram)"
    print(f"P({args.word!r} {ctx_str}) = {prob:.6f}  [log2 = {log_prob:.4f}]")


def cmd_demo(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("KILGARRIFF CORPUS DEMO")
    print("=" * 60)
    cmd_train(argparse.Namespace(corpus="kilgarriff", n=3, output=None))
    cmd_generate(
        argparse.Namespace(
            model=str(_MODELS_DIR / "kilgarriff_ngram_model.pkl"),
            words=20,
            seed=7,
            count=3,
        )
    )

    print()
    print("=" * 60)
    print("TRUMP TWEETS CORPUS DEMO")
    print("=" * 60)
    cmd_train(argparse.Namespace(corpus="trump", n=3, output=None))
    cmd_generate(
        argparse.Namespace(
            model=str(_MODELS_DIR / "trump_ngram_model.pkl"),
            words=40,
            seed=42,
            count=3,
        )
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ngram-lm",
        description="N-gram Language Models — pure Python",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a model and save to disk")
    p_train.add_argument(
        "corpus",
        choices=["kilgarriff", "trump"],
        help="Which corpus to train on",
    )
    p_train.add_argument(
        "--n", type=int, default=3, metavar="N",
        help="N-gram order (default: 3)",
    )
    p_train.add_argument(
        "--output", "-o", default=None, metavar="PATH",
        help="Output .pkl path (default: models/<corpus>_ngram_model.pkl)",
    )
    p_train.set_defaults(func=cmd_train)

    # generate
    p_gen = sub.add_parser("generate", help="Generate text from a saved model")
    p_gen.add_argument("model", metavar="MODEL_PATH", help="Path to .pkl file")
    p_gen.add_argument(
        "--words", "-w", type=int, default=20,
        help="Max tokens to generate (default: 20)",
    )
    p_gen.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p_gen.add_argument(
        "--count", "-c", type=int, default=1,
        help="Number of sentences to generate using consecutive seeds",
    )
    p_gen.set_defaults(func=cmd_generate)

    # score
    p_score = sub.add_parser("score", help="Score a word probability")
    p_score.add_argument("model", metavar="MODEL_PATH", help="Path to .pkl file")
    p_score.add_argument("word", help="Word to score")
    p_score.add_argument(
        "--context", "-c", nargs="+", default=None,
        help="Left context words (e.g. --context language is)",
    )
    p_score.set_defaults(func=cmd_score)

    # demo
    p_demo = sub.add_parser("demo", help="Run the full end-to-end demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
