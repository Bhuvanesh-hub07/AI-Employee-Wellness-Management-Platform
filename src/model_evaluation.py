import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/bert_heldout_test.csv"
OUTPUT_FILE = "outputs/model_comparison.csv"

MAX_LENGTH = 64

MODELS = {
    "BERT": "models/bert_emotion",
    "DistilBERT": "models/distilbert_emotion"
}


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path):
    """
    Load tokenizer and trained sequence classification model.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


# ============================================================
# PREDICTION
# ============================================================

def predict(texts, tokenizer, model):
    """
    Generate emotion predictions for multiple texts.
    """

    predictions = []

    for text in texts:

        if not isinstance(text, str):
            text = str(text)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        prediction_id = torch.argmax(
            outputs.logits,
            dim=-1
        ).item()

        emotion = model.config.id2label[
            prediction_id
        ]

        predictions.append(
            emotion
        )

    return predictions


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    texts,
    labels
):
    """
    Evaluate one model using:

        Accuracy
        Macro Precision
        Macro Recall
        Macro F1

    Also displays classification report
    and confusion matrix.
    """

    print(
        "\n" + "=" * 75
    )

    print(
        f"{model_name} EVALUATION"
    )

    print(
        "=" * 75
    )

    tokenizer, model = load_model(
        model_path
    )

    predictions = predict(
        texts,
        tokenizer,
        model
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        f"Accuracy     : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision    : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall       : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Macro F1     : "
        f"{macro_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            labels,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels_order = sorted(
        set(labels)
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=labels_order
    )

    print(
        "Confusion Matrix:"
    )

    print(
        f"Labels: {labels_order}"
    )

    print(
        matrix
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "TASK 5 - BERT VS DISTILBERT MODEL EVALUATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        TEST_FILE
    )

    required_columns = {
        "text",
        "label"
    }

    if not required_columns.issubset(
        df.columns
    ):
        raise ValueError(
            "Test file must contain "
            "'text' and 'label' columns."
        )

    df = df.dropna(
        subset=["text", "label"]
    )

    texts = (
        df["text"]
        .astype(str)
        .tolist()
    )

    labels = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
        .tolist()
    )

    if not texts:
        raise ValueError(
            "Test dataset contains no valid text samples."
        )

    print(
        f"\nEvaluation samples: {len(texts)}"
    )

    # --------------------------------------------------------
    # Evaluate all configured models
    # --------------------------------------------------------

    results = []

    for model_name, model_path in (
        MODELS.items()
    ):

        result = evaluate_model(
            model_name,
            model_path,
            texts,
            labels
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 75
    )

    display_df = results_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "Macro F1"
    ]

    for column in metric_columns:

        display_df[column] = (
            display_df[column]
            * 100
        ).round(2).astype(str) + "%"

    print(
        display_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Identify best models dynamically
    # --------------------------------------------------------

    best_accuracy = results_df.loc[
        results_df["Accuracy"].idxmax(),
        "Model"
    ]

    best_f1 = results_df.loc[
        results_df["Macro F1"].idxmax(),
        "Model"
    ]

    print(
        "\n" + "=" * 75
    )

    print(
        "TASK 5 VALIDATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Best Accuracy Model : "
        f"{best_accuracy}"
    )

    print(
        f"Best Macro F1 Model  : "
        f"{best_f1}"
    )

    print(
        "\nComparison saved to:"
        f" {OUTPUT_FILE}"
    )