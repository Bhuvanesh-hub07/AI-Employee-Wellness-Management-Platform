"""
Integrated AI Employee Wellness text-analysis pipeline.

Flow:
Input
  -> Validation
  -> Preprocessing
  -> VADER Sentiment
  -> BERT Emotion Classification
  -> Confidence Validation
  -> Multi-Label Interpretation
  -> Final Result

This module integrates Milestone 1 and Milestone 2.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ingestion import (
    ingest_csv,
    ingest_direct_text,
    ingest_txt,
)
from preprocessing import preprocess_text
from sentiment import analyze_sentiment


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BERT_MODEL_PATH = "models/bert_emotion"

MAX_LENGTH = 64

# Multi-label interpretation thresholds.
MULTI_LABEL_THRESHOLD = 0.17
RELATIVE_TO_PRIMARY = 0.90

# Confidence thresholds.
HIGH_CONFIDENCE = 0.30
MODERATE_CONFIDENCE = 0.22
LOW_MARGIN = 0.03
VERY_LOW_MARGIN = 0.015


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_bert_model():
    """Load the BERT tokenizer and model once."""

    print("\nLoading BERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_PATH
    )

    model.eval()

    print("BERT model loaded successfully.\n")

    return tokenizer, model


# Load the model once when the pipeline starts.
TOKENIZER, MODEL = load_bert_model()


# ---------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------

def determine_confidence_status(
    confidence,
    margin,
):
    """
    Determine confidence level using:

    1. Top prediction probability.
    2. Difference between top-1 and top-2 predictions.
    """

    if (
        confidence >= HIGH_CONFIDENCE
        and margin >= LOW_MARGIN
    ):
        return "HIGH CONFIDENCE"

    if (
        confidence >= MODERATE_CONFIDENCE
        and margin >= LOW_MARGIN
    ):
        return "MODERATE CONFIDENCE"

    if margin < VERY_LOW_MARGIN:
        return "VERY UNCERTAIN"

    return "LOW CONFIDENCE / UNCERTAIN"


# ---------------------------------------------------------------------
# Emotion analysis
# ---------------------------------------------------------------------

def analyze_emotion(text):
    """
    Perform BERT emotion classification.

    Returns:
        Primary emotion.
        Confidence.
        Second-best emotion.
        Confidence margin.
        Confidence status.
        Secondary emotions.
        Complete probability distribution.
    """

    if not isinstance(text, str):
        raise ValueError(
            "Text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Text cannot be empty."
        )

    # Tokenization.
    inputs = TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    # BERT prediction.
    with torch.no_grad():
        outputs = MODEL(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    # Convert probabilities to a label dictionary.
    id2label = MODEL.config.id2label

    probability_map = {
        id2label[index]: probabilities[index].item()
        for index in range(len(probabilities))
    }

    # Rank emotions by probability.
    ranked_emotions = sorted(
        probability_map.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    primary_emotion = ranked_emotions[0][0]
    primary_confidence = ranked_emotions[0][1]

    second_emotion = ranked_emotions[1][0]
    second_confidence = ranked_emotions[1][1]

    # Confidence margin.
    confidence_margin = (
        primary_confidence
        - second_confidence
    )

    confidence_status = determine_confidence_status(
        primary_confidence,
        confidence_margin,
    )

    # -----------------------------------------------------------------
    # Multi-label interpretation
    # -----------------------------------------------------------------
    #
    # NOTE:
    # This is a heuristic interpretation of a single-label
    # softmax model. It is not a true multi-label classifier.
    #

    secondary_emotions = []

    for emotion, probability in ranked_emotions[1:]:

        meets_absolute_threshold = (
            probability >= MULTI_LABEL_THRESHOLD
        )

        meets_relative_threshold = (
            probability
            >= primary_confidence * RELATIVE_TO_PRIMARY
        )

        if (
            meets_absolute_threshold
            and meets_relative_threshold
        ):
            secondary_emotions.append(
                emotion
            )

    detected_emotions = [
        primary_emotion
    ] + secondary_emotions

    # Determine analysis type.
    if len(detected_emotions) == 1:
        analysis_type = "Single emotion"

    elif len(detected_emotions) == 2:
        analysis_type = "Two-emotion combination"

    else:
        analysis_type = "Mixed emotion"

    return {
        "emotion": primary_emotion,
        "confidence": primary_confidence,
        "second_best_emotion": second_emotion,
        "second_best_confidence": second_confidence,
        "confidence_margin": confidence_margin,
        "confidence_status": confidence_status,
        "secondary_emotions": secondary_emotions,
        "detected_emotions": detected_emotions,
        "emotion_count": len(detected_emotions),
        "analysis_type": analysis_type,
        "probabilities": probability_map,
    }


# ---------------------------------------------------------------------
# Complete text pipeline
# ---------------------------------------------------------------------

def process_text(text):
    """
    Process text through the complete Milestone 1 + Milestone 2
    wellness analysis pipeline.

    Flow:
        Input validation
        -> Preprocessing
        -> VADER sentiment
        -> BERT emotion classification
        -> Confidence validation
        -> Multi-label interpretation
        -> Final result
    """

    # Step 1: Input validation.
    valid_text = ingest_direct_text(
        text
    )

    # Step 2: Text preprocessing.
    preprocessing_result = preprocess_text(
        valid_text
    )

    # Step 3: VADER sentiment analysis.
    sentiment_result = analyze_sentiment(
        valid_text
    )

    # Step 4-6: BERT emotion, confidence,
    # and multi-label interpretation.
    emotion_result = analyze_emotion(
        valid_text
    )

    # Step 7: Integrated final result.
    return {
        # Original and processed text.
        "original_text": valid_text,
        "processed_text": (
            preprocessing_result[
                "processed_text"
            ]
        ),

        # Milestone 1 - Sentiment.
        "sentiment": (
            sentiment_result[
                "classification"
            ]
        ),
        "compound": sentiment_result["compound"],
        "positive": sentiment_result["positive"],
        "negative": sentiment_result["negative"],
        "neutral": sentiment_result["neutral"],

        # Milestone 2 - Primary emotion.
        "emotion": emotion_result["emotion"],
        "confidence": emotion_result["confidence"],

        # Milestone 2 - Confidence validation.
        "second_best_emotion": (
            emotion_result[
                "second_best_emotion"
            ]
        ),
        "second_best_confidence": (
            emotion_result[
                "second_best_confidence"
            ]
        ),
        "confidence_margin": (
            emotion_result[
                "confidence_margin"
            ]
        ),
        "confidence_status": (
            emotion_result[
                "confidence_status"
            ]
        ),

        # Milestone 2 - Multi-label interpretation.
        "secondary_emotions": (
            emotion_result[
                "secondary_emotions"
            ]
        ),
        "detected_emotions": (
            emotion_result[
                "detected_emotions"
            ]
        ),
        "emotion_count": (
            emotion_result[
                "emotion_count"
            ]
        ),
        "analysis_type": (
            emotion_result[
                "analysis_type"
            ]
        ),

        # Complete probability distribution.
        "probabilities": (
            emotion_result[
                "probabilities"
            ]
        ),
    }


# ---------------------------------------------------------------------
# TXT file processing
# ---------------------------------------------------------------------

def process_txt_file(file_path):
    """Process a TXT file through the integrated pipeline."""

    text = ingest_txt(
        file_path
    )

    return process_text(
        text
    )


# ---------------------------------------------------------------------
# CSV file processing
# ---------------------------------------------------------------------

def process_csv_file(file_path):
    """Process all text entries from a CSV file."""

    texts = ingest_csv(
        file_path
    )

    results = []

    for text in texts:
        results.append(
            process_text(text)
        )

    return results


# ---------------------------------------------------------------------
# Pipeline test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print(
        "MILESTONE 1 + MILESTONE 2 "
        "INTEGRATED PIPELINE"
    )
    print("=" * 80)

    test_cases = [
        "I am very happy with my project.",
        "The workload is stressful and unfair.",
        "I am nervous about my performance review.",
        "I feel proud of my team's achievement.",
        "I am frustrated with my workload but excited about my promotion.",
    ]

    for number, text in enumerate(
        test_cases,
        start=1,
    ):

        print("\n" + "-" * 80)
        print(f"TEST {number}")
        print("-" * 80)

        result = process_text(
            text
        )

        print(
            f"Text       : "
            f"{result['original_text']}"
        )

        print(
            f"Sentiment  : "
            f"{result['sentiment']}"
        )

        print(
            f"Compound   : "
            f"{result['compound']:.4f}"
        )

        print(
            f"Emotion    : "
            f"{result['emotion']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence']:.2%}"
        )

        print(
            f"Second Best: "
            f"{result['second_best_emotion']} "
            f"({result['second_best_confidence']:.2%})"
        )

        print(
            f"Margin     : "
            f"{result['confidence_margin']:.2%}"
        )

        print(
            f"Status     : "
            f"{result['confidence_status']}"
        )

        print(
            f"Detected   : "
            f"{', '.join(result['detected_emotions'])}"
        )

        print(
            f"Emotion Count: "
            f"{result['emotion_count']}"
        )

        print(
            f"Analysis   : "
            f"{result['analysis_type']}"
        )

        print("\nProbabilities:")

        for emotion, probability in sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"  {emotion:<10}: "
                f"{probability:.2%}"
            )

    print("\n" + "=" * 80)
    print(
        "INTEGRATED PIPELINE TEST COMPLETE"
    )
    print("=" * 80)