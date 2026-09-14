"""
ISEAR held-out benchmark evaluation.

Evaluates the trained BERT and DistilBERT emotion classifiers
on the ISEAR held-out dataset and compares their performance.

Note:
The original ISEAR dataset contains seven emotion categories.
This project evaluates the five categories supported by the
current models:
    joy, sadness, anger, fear, disgust

Shame and guilt are excluded from this evaluation.
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_FILE = Path("data/isear_heldout_test.csv")
OUTPUT_DIR = Path("outputs")
MAX_LENGTH = 64

MODELS = {
    "BERT": Path("models/bert_emotion"),
    "DistilBERT": Path("models/distilbert_emotion"),
}

SUPPORTED_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
]


# ---------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------

def load_model(model_path):
    """Load a tokenizer and trained sequence-classification model."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


def predict_with_confidence(texts, tokenizer, model):
    """
    Predict emotions and confidence scores for a list of texts.

    Returns:
        predictions: Predicted emotion for each text.
        confidences: Softmax confidence of the predicted emotion.
        all_probabilities: Probability distribution for every emotion.
    """

    predictions = []
    confidences = []
    all_probabilities = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        prediction_index = torch.argmax(
            probabilities
        ).item()

        predicted_emotion = model.config.id2label[
            prediction_index
        ]

        confidence = probabilities[
            prediction_index
        ].item()

        probability_dict = {
            model.config.id2label[index]: round(
                probabilities[index].item(),
                4,
            )
            for index in range(len(probabilities))
        }

        predictions.append(predicted_emotion)
        confidences.append(confidence)
        all_probabilities.append(probability_dict)

    return predictions, confidences, all_probabilities


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(
    model_name,
    model_path,
    texts,
    true_labels,
):
    """Evaluate one model on the ISEAR held-out dataset."""

    print("\n" + "=" * 80)
    print(f"{model_name} - ISEAR VALIDATION")
    print("=" * 80)

    tokenizer, model = load_model(model_path)

    (
        predictions,
        confidences,
        probabilities,
    ) = predict_with_confidence(
        texts,
        tokenizer,
        model,
    )

    # Overall metrics
    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    precision = precision_score(
        true_labels,
        predictions,
        labels=SUPPORTED_EMOTIONS,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        labels=SUPPORTED_EMOTIONS,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=SUPPORTED_EMOTIONS,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Precision    : {precision * 100:.2f}%")
    print(f"Recall       : {recall * 100:.2f}%")
    print(f"Macro F1     : {macro_f1 * 100:.2f}%")

    # Emotion-wise performance
    print("\nEmotion-wise Performance:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=SUPPORTED_EMOTIONS,
            zero_division=0,
        )
    )

    # Confusion matrix
    print("Confusion Matrix:")

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=SUPPORTED_EMOTIONS,
    )

    print("Labels:", SUPPORTED_EMOTIONS)
    print(matrix)

    # Prediction statistics
    average_confidence = sum(confidences) / len(confidences)

    correct_count = sum(
        actual == predicted
        for actual, predicted in zip(
            true_labels,
            predictions,
        )
    )

    incorrect_count = len(true_labels) - correct_count

    print(
        f"\nCorrect Predictions   : {correct_count}"
    )

    print(
        f"Incorrect Predictions : {incorrect_count}"
    )

    print(
        f"Average Confidence    : "
        f"{average_confidence * 100:.2f}%"
    )

    # -----------------------------------------------------------------
    # Save detailed predictions
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        {
            "text": texts,
            "expected_label": true_labels,
            "predicted_label": predictions,
            "confidence": confidences,
        }
    )

    results_df["correct"] = (
        results_df["expected_label"]
        == results_df["predicted_label"]
    )

    for emotion in SUPPORTED_EMOTIONS:
        results_df[f"prob_{emotion}"] = [
            probabilities_for_text.get(emotion, 0)
            for probabilities_for_text in probabilities
        ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"{model_name.lower()}_isear_evaluation.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nDetailed results saved to: {output_file}"
    )

    # -----------------------------------------------------------------
    # Display incorrect predictions
    # -----------------------------------------------------------------

    incorrect = results_df[
        ~results_df["correct"]
    ]

    print("\nFirst 10 Incorrect Predictions:")

    if incorrect.empty:
        print("None")
    else:
        for _, row in incorrect.head(10).iterrows():
            print(
                f"\nExpected   : {row['expected_label']}"
            )
            print(
                f"Predicted  : {row['predicted_label']}"
            )
            print(
                f"Confidence : "
                f"{row['confidence'] * 100:.2f}%"
            )
            print(
                f"Text       : {row['text']}"
            )

    return {
        "Model": model_name,
        "Samples": len(true_labels),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1,
        "Correct": correct_count,
        "Incorrect": incorrect_count,
        "Average Confidence": average_confidence,
    }


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------

def load_iseAr_dataset():
    """Load and validate the ISEAR held-out dataset."""

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"ISEAR test file not found: {TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "ISEAR file must contain 'text' and 'label' columns."
        )

    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    df["text"] = df["text"].astype(str)

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Keep only emotions supported by the current models.
    df = df[
        df["label"].isin(SUPPORTED_EMOTIONS)
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No supported ISEAR samples were found."
        )

    return df


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    """Run the complete ISEAR benchmark evaluation."""

    print("=" * 80)
    print("TASK 6 - ISEAR HELD-OUT BENCHMARK VALIDATION")
    print("=" * 80)

    df = load_iseAr_dataset()

    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    print(
        f"\nISEAR samples evaluated: {len(df)}"
    )

    print("\nEmotion distribution:")
    print(
        df["label"].value_counts()
    )

    results = []

    for model_name, model_path in MODELS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_name} model not found: {model_path}"
            )

        result = evaluate_model(
            model_name,
            model_path,
            texts,
            true_labels,
        )

        results.append(result)

    # -----------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------

    comparison_df = pd.DataFrame(results)

    comparison_file = (
        OUTPUT_DIR / "isear_model_comparison.csv"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    print("\n" + "=" * 80)
    print("ISEAR MODEL COMPARISON")
    print("=" * 80)

    display_df = comparison_df.copy()

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "Macro F1",
        "Average Confidence",
    ]

    for column in percentage_columns:
        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"

    print(
        display_df.to_string(
            index=False
        )
    )

    # Determine best models dynamically.
    best_accuracy = comparison_df.loc[
        comparison_df["Accuracy"].idxmax(),
        "Model",
    ]

    best_f1 = comparison_df.loc[
        comparison_df["Macro F1"].idxmax(),
        "Model",
    ]

    print(
        f"\nBest Accuracy Model : {best_accuracy}"
    )

    print(
        f"Best Macro F1 Model : {best_f1}"
    )

    print(
        f"\nComparison saved to: {comparison_file}"
    )

    print("\n" + "=" * 80)
    print("TASK 6 ISEAR VALIDATION COMPLETE")
    print("=" * 80)

    print(
        "\nNote:"
        "\nISEAR originally contains seven emotion categories."
        "\nThis evaluation uses only the five categories"
        "\ndirectly supported by the current models:"
        "\njoy, sadness, anger, fear, disgust."
        "\nShame and guilt are excluded."
    )


if __name__ == "__main__":
    main()