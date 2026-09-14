"""
Inspect emotion probability distributions from the trained DistilBERT model.

This utility displays the softmax probability assigned to each
supported emotion for a set of representative workplace texts.
"""

from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path("models/distilbert_emotion")
MAX_LENGTH = 64

TEST_SENTENCES = [
    "I am very happy and proud of my work.",
    "I feel sad and emotionally exhausted.",
    "I am angry and frustrated about my workload.",
    "I am anxious and worried about the deadline.",
    "I was surprised by the unexpected recognition.",
    "I dislike the toxic behavior in my workplace.",
    "I am anxious about the deadline and frustrated by my workload.",
    "I feel disappointed, stressed, and unsupported by my team.",
    "I am happy with my achievement but nervous about the next project.",
    "I am frustrated with the workload but excited about the new opportunity.",
]


# ---------------------------------------------------------------------
# Probability inspection
# ---------------------------------------------------------------------

def inspect_probabilities(
    model_path=MODEL_PATH,
):
    """
    Display ranked emotion probabilities for test sentences.

    Args:
        model_path: Path to the trained DistilBERT model.
    """

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"DistilBERT model not found: {model_path}"
        )

    print("Loading DistilBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    id2label = model.config.id2label

    for number, text in enumerate(
        TEST_SENTENCES,
        start=1,
    ):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        ranked = sorted(
            (
                (
                    id2label[index],
                    probability.item(),
                )
                for index, probability in enumerate(
                    probabilities
                )
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        print("\n" + "=" * 75)
        print(f"Sample {number}")
        print("=" * 75)

        print("Text:")
        print(text)

        print("\nEmotion probabilities:")

        for emotion, probability in ranked:
            print(
                f"{emotion:<10} "
                f"{probability:.4f} "
                f"({probability:.2%})"
            )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    inspect_probabilities()