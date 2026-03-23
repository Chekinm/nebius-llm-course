"""Model serialization using stdlib pickle.

Unlike the NLTK-based variant, the custom NgramLM class contains no lambda
functions, so standard pickle works without needing the dill library.
"""

import pathlib
import pickle

from ngram_lm.model import NgramLM


def save_model(model: NgramLM, path: str | pathlib.Path) -> None:
    """Serialize a fitted NgramLM model to disk.

    Args:
        model: A fitted NgramLM model.
        path: Destination file path (conventionally *.pkl).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        pickle.dump(model, fout)
    print(f"Model saved to {path}")


def load_model(path: str | pathlib.Path) -> NgramLM:
    """Load a previously serialized NgramLM model from disk.

    Args:
        path: Path to the .pkl file.

    Returns:
        The deserialized NgramLM model.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No model file found at {path}.\n"
            "Run 'train' first to create a model."
        )
    with path.open("rb") as fin:
        return pickle.load(fin)
