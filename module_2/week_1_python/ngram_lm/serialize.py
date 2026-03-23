"""Model serialization using dill (required for NLTK MLE lambdas).

Standard pickle cannot serialize nltk.lm.MLE models because they
contain lambda functions internally. This module uses dill as a drop-in
replacement, making that detail invisible to callers.
"""

import pathlib

import dill
from nltk.lm import MLE


def save_model(model: MLE, path: str | pathlib.Path) -> None:
    """Serialize a fitted MLE model to disk.

    Args:
        model: A fitted MLE model.
        path: Destination file path (conventionally *.pkl).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        dill.dump(model, fout)
    print(f"Model saved to {path}")


def load_model(path: str | pathlib.Path) -> MLE:
    """Load a previously serialized MLE model from disk.

    Args:
        path: Path to the .pkl file.

    Returns:
        The deserialized MLE model.

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
        return dill.load(fin)
