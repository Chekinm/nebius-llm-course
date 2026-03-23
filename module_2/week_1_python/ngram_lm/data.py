"""Data loading utilities.

All paths to pre-existing data files are resolved relative to this file,
pointing at ../week_1/ so no data needs to be duplicated.
"""

import io
import pathlib

import pandas as pd
import requests

# Resolve sibling week_1 directory relative to this file's location:
#   .../module_2/week_1_python/ngram_lm/data.py
#   -> .../module_2/week_1/
_WEEK1_DIR = pathlib.Path(__file__).parent.parent.parent / "week_1"

KILGARRIFF_LOCAL = _WEEK1_DIR / "language-never-random.txt"
KILGARRIFF_URL = (
    "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a"
    "/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
)
TRUMP_CSV = _WEEK1_DIR / "Donald-Tweets!.csv"


def load_kilgarriff() -> str:
    """Return the full Kilgarriff text as a string.

    Tries the local cached file first; falls back to HTTP download.
    The downloaded file is cached at the week_1/ location for future runs.
    """
    if KILGARRIFF_LOCAL.is_file():
        return KILGARRIFF_LOCAL.read_text(encoding="utf-8")

    print(f"Downloading Kilgarriff text from {KILGARRIFF_URL} ...")
    response = requests.get(KILGARRIFF_URL, timeout=30)
    response.raise_for_status()
    text = response.content.decode("utf-8")
    KILGARRIFF_LOCAL.write_text(text, encoding="utf-8")
    print(f"Cached to {KILGARRIFF_LOCAL}")
    return text


def load_trump_tweets() -> list[str]:
    """Return a list of raw tweet strings from the Trump CSV dataset."""
    if not TRUMP_CSV.is_file():
        raise FileNotFoundError(
            f"Trump tweets CSV not found at {TRUMP_CSV}.\n"
            "Place Donald-Tweets!.csv in module_2/week_1/ to use this corpus."
        )
    df = pd.read_csv(TRUMP_CSV)
    return df["Tweet_Text"].dropna().tolist()
