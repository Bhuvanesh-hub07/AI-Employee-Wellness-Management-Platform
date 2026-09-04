from pathlib import Path
import pandas as pd


def validate_text(text):
    """Validate and clean a single text input."""

    if text is None:
        raise ValueError("Text input is empty.")

    if not isinstance(text, str):
        raise ValueError("Text input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text input is empty.")

    return text


def ingest_direct_text(text):
    """Read text entered directly by the user."""
    return validate_text(text)


def ingest_txt(file_path):
    """Read text from a TXT file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported.")

    text = path.read_text(encoding="utf-8")

    return validate_text(text)


def ingest_csv(file_path, text_column="text"):
    """Read multiple text records from a CSV file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")

    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(
            f"CSV must contain a '{text_column}' column."
        )

    texts = []

    for value in df[text_column]:

        if pd.notna(value):
            value = str(value).strip()

            if value:
                texts.append(value)

    if not texts:
        raise ValueError("CSV contains no valid text data.")

    return texts