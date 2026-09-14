"""
Independent held-out evaluation for the trained BERT emotion classifier.

This evaluation uses a separate held-out dataset to measure:
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix
- Prediction confidence
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path("models/bert_emotion")
TEST_PATH = Path("data/bert_heldout_test.csv")
OUTPUT_PATH = Path("outputs/bert_heldout_evaluation.csv")

MAX_LENGTH = 64


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_heldout(
    test_path=TEST_PATH,
    model_path=MODEL_PATH,
):
    """Evaluate BERT on the independent held-out dataset."""

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"BERT model not found: {model_path}"
        )

    if not Path(test_path).exists():
        raise FileNotFoundError(
            f"Held-out dataset not found: {test_path}"
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
            "Held-out dataset must contain "
            "'text' and 'label' columns."
        )

    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[df["text"] != ""]

    if df.empty:
        raise ValueError(
            "No valid held-out samples were found."
        )

    id2label = model.config.id2label
    labels = list(model.config.label2id.keys())

    results = []

    # -----------------------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------------------

    for _, row in df.iterrows():

        text = row["text"]
        expected = row["label"]

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

        predicted_emotion = id2label[
            predicted_id
        ]

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

    results_df = pd.DataFrame(results)

    y_true = results_df["expected"]
    y_pred = results_df["predicted"]

    # -----------------------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # -----------------------------------------------------------------
    # Display sample-by-sample results
    # -----------------------------------------------------------------

    print("\n" + "=" * 75)
    print("INDEPENDENT BERT HELD-OUT EVALUATION")
    print("=" * 75)

    for index, row in results_df.iterrows():

        status = (
            "PASS"
            if row["correct"]
            else "FAIL"
        )

        print(f"\nSample {index + 1}")
        print(f"Text       : {row['text']}")
        print(f"Expected   : {row['expected']}")
        print(f"Predicted  : {row['predicted']}")
        print(f"Confidence : {row['confidence']:.2%}")
        print(f"Result     : {status}")

    # -----------------------------------------------------------------
    # Display metrics
    # -----------------------------------------------------------------

    print("\n" + "=" * 75)
    print("INDEPENDENT TEST METRICS")
    print("=" * 75)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Macro F1  : {f1:.4f}")

    print("\nConfusion Matrix")
    print("Labels:", labels)
    print(matrix)

    # -----------------------------------------------------------------
    # Save results
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
        f"\nHeld-out evaluation saved to: "
        f"{OUTPUT_PATH}"
    )

    return results_df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_heldout()