import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"
MODEL_OUTPUT_DIR = "models/bert_emotion"

MAX_LENGTH = 64
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


# ============================================================
# EMOTION LABELS
# ============================================================

LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for index, label in enumerate(LABELS)
}


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(csv_path):
    """
    Load and validate an emotion classification dataset.

    Required columns:
        text
        label
    """

    df = pd.read_csv(csv_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    # Remove rows with missing text or labels
    df = df.dropna(
        subset=["text", "label"]
    )

    # Normalize text and labels
    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Remove empty text
    df = df[df["text"] != ""]

    # Validate emotion labels
    invalid_labels = (
        set(df["label"]) - set(LABELS)
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid emotion labels found: {invalid_labels}"
        )

    # Convert labels to numeric IDs
    df["label_id"] = df["label"].map(LABEL2ID)

    # Hugging Face Trainer expects the target column
    # to be named "labels"
    return Dataset.from_pandas(
        df[["text", "label_id"]].rename(
            columns={"label_id": "labels"}
        ),
        preserve_index=False
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize a Hugging Face dataset for BERT.
    """

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    return dataset.map(
        tokenize_function,
        batched=True
    )


# ============================================================
# MODEL EVALUATION METRICS
# ============================================================

def compute_metrics(eval_prediction):
    """
    Calculate classification metrics:

    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1
    """

    predictions, labels = eval_prediction

    predicted_labels = predictions.argmax(
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    }


# ============================================================
# BERT TRAINING
# ============================================================

def train_bert(
    train_path,
    test_path,
    output_dir=MODEL_OUTPUT_DIR
):
    """
    Fine-tune BERT for six-emotion classification.
    """

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading datasets...")

    train_dataset = load_dataset(
        train_path
    )

    test_dataset = load_dataset(
        test_path
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # Tokenize datasets
    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer
    )

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer
    )

    print("Loading pretrained BERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=5,
        report_to="none",
        save_total_limit=1,
        dataloader_pin_memory=False
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print(
        "\nStarting BERT training...\n"
    )

    trainer.train()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating BERT model...\n"
    )

    metrics = trainer.evaluate()

    print(
        "\nEvaluation Results:"
    )

    for key, value in metrics.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.4f}"
            )

        else:
            print(
                f"{key}: {value}"
            )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print(
        "\nSaving BERT model..."
    )

    trainer.save_model(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    print(
        f"BERT model saved to: {output_dir}"
    )

    return trainer, tokenizer


# ============================================================
# BERT PREDICTION
# ============================================================

def predict_emotion(
    text,
    model_path=MODEL_OUTPUT_DIR
):
    """
    Predict the primary emotion for a text input.

    Returns:
        emotion
        confidence
        probability distribution
    """

    if not isinstance(text, str):
        raise ValueError(
            "Text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Text must be a non-empty string."
        )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Generate prediction
    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # Primary prediction
    predicted_id = torch.argmax(
        probabilities
    ).item()

    emotion = ID2LABEL[
        predicted_id
    ]

    confidence = probabilities[
        predicted_id
    ].item()

    # Complete probability distribution
    emotion_probabilities = {
        ID2LABEL[index]: round(
            probability.item(),
            4
        )
        for index, probability in enumerate(
            probabilities
        )
    }

    return {
        "text": text,
        "emotion": emotion,
        "confidence": round(
            confidence,
            4
        ),
        "probabilities": emotion_probabilities
    }


# ============================================================
# MAIN TRAINING AND SAMPLE PREDICTION
# ============================================================

if __name__ == "__main__":

    train_path = (
        "data/emotion_train_expanded.csv"
    )

    test_path = (
        "data/emotion_test.csv"
    )

    # Train BERT
    trainer, tokenizer = train_bert(
        train_path,
        test_path
    )

    # Sample prediction
    sample_text = (
        "I am very happy and excited "
        "about my current project."
    )

    result = predict_emotion(
        sample_text
    )

    print(
        "\nSample Prediction:"
    )

    print(
        "Text:",
        result["text"]
    )

    print(
        "Emotion:",
        result["emotion"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print(
        "\nEmotion Probabilities:"
    )

    for emotion, probability in (
        result["probabilities"].items()
    ):

        print(
            f"{emotion}: {probability}"
        )