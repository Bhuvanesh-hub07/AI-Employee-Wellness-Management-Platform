import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Load stop words
STOP_WORDS = set(stopwords.words("english"))

# Keep words that can affect sentiment
IMPORTANT_WORDS = {
    "not",
    "no",
    "never",
    "nor",
    "very",
    "too",
    "really",
    "hardly",
    "barely"
}

# Remove only stop words that are not important for sentiment
STOP_WORDS = STOP_WORDS - IMPORTANT_WORDS

lemmatizer = WordNetLemmatizer()


def normalize_text(text):
    """Normalize spaces and validate text."""

    if text is None:
        raise ValueError("Text cannot be None.")

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text is empty.")

    # Replace repeated spaces/newlines/tabs
    text = re.sub(r"\s+", " ", text)

    return text


def remove_noise(text):
    """Remove URLs and HTML-like content."""

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Clean repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """Convert text into tokens."""

    return word_tokenize(text)


def remove_stopwords(tokens):
    """Remove stop words while preserving sentiment-important words."""

    return [
        token
        for token in tokens
        if token.lower() not in STOP_WORDS
    ]


def lemmatize_tokens(tokens):
    """Convert words to their base form."""

    return [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
    ]


def preprocess_text(text):
    """Run the complete preprocessing pipeline."""

    original_text = text

    normalized_text = normalize_text(text)

    cleaned_text = remove_noise(normalized_text)

    if not cleaned_text:
        raise ValueError(
            "Text became empty after noise filtering."
        )

    tokens = tokenize_text(cleaned_text)

    filtered_tokens = remove_stopwords(tokens)

    lemmatized_tokens = lemmatize_tokens(filtered_tokens)

    processed_text = " ".join(lemmatized_tokens)

    if not processed_text.strip():
        raise ValueError("Processed text is empty.")

    return {
        "original_text": original_text,
        "normalized_text": normalized_text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "processed_text": processed_text
    }