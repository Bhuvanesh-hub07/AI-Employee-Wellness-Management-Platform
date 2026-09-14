"""
Sample-by-sample evaluation for the trained BERT emotion classifier.

This script evaluates the trained BERT model on the configured
emotion test dataset and saves a detailed prediction report.
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path("models/bert_emotion")
TEST_PATH = Path("data/emotion_test.csv")
OUTPUT_PATH = Path("outputs/bert_sample_evaluation.csv")

MAX_LENGTH = 64

LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_bert(
    test_path=TEST_PATH,
    model_path=MODEL_PATH,
):
    """
    Evaluate the trained BERT model sample by sample.

    Args:
        test_path: Path to the test dataset.
        model_path: Path to the trained BERT model.

    Returns:
        pandas.DataFrame containing detailed predictions.
    """

    if not Path(test_path).exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"BERT model not found: {model_path}"
        )

    print("Loading trained BERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    df = pd.read_csv(test_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Test dataset must contain 'text' and 'label' columns."
        )

    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    results = []

    for _, row in df.iterrows():

        text = str(row["text"]).strip()
        expected = str(row["label"]).strip().lower()

        if not text:
            continue

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

        predicted_id = torch.argmax(
            probabilities
        ).item()

        if predicted_id >= len(LABELS):
            raise ValueError(
                f"Model predicted an unknown label index: "
                f"{predicted_id}"
            )

        predicted_emotion = LABELS[predicted_id]

        confidence = probabilities[
            predicted_id
        ].item()

        results.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted_emotion,
                "confidence": round(
                    confidence,
                    4,
                ),
                "correct": (
                    predicted_emotion == expected
                ),
            }
        )

    if not results:
        raise ValueError(
            "No valid test samples were available for evaluation."
        )

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------------------
    # Display sample-by-sample results
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BERT SAMPLE-BY-SAMPLE EVALUATION")
    print("=" * 70)

    for index, row in results_df.iterrows():

        status = (
            "PASS"
            if row["correct"]
            else "FAIL"
        )

        print(f"\nSample {index + 1}:")
        print(f"Text       : {row['text']}")
        print(f"Expected   : {row['expected']}")
        print(f"Predicted  : {row['predicted']}")
        print(f"Confidence : {row['confidence']:.2%}")
        print(f"Result     : {status}")

    # -----------------------------------------------------------------
    # Accuracy
    # -----------------------------------------------------------------

    accuracy = results_df["correct"].mean()

    print("\n" + "=" * 70)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Save evaluation report
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nEvaluation report saved to: "
        f"{OUTPUT_PATH}"
    )

    return results_df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_bert()