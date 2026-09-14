import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/distilbert_emotion"
MAX_LENGTH = 64


# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

HIGH_CONFIDENCE = 0.30
MODERATE_CONFIDENCE = 0.22

LOW_MARGIN = 0.03
VERY_LOW_MARGIN = 0.015


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path=MODEL_PATH):
    """
    Load the tokenizer and trained emotion classification model.
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
# CONFIDENCE STATUS
# ============================================================

def determine_uncertainty(confidence, margin):
    """
    Determine confidence status using:

    1. Top prediction probability
    2. Difference between the top two predictions

    Returns:
        HIGH CONFIDENCE
        MODERATE CONFIDENCE
        LOW CONFIDENCE / UNCERTAIN
        VERY UNCERTAIN
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


# ============================================================
# CONFIDENCE VALIDATION
# ============================================================

def validate_confidence(
    text,
    tokenizer,
    model
):
    """
    Analyze model confidence for a text input.

    Returns:
        Primary emotion
        Confidence
        Second-best emotion
        Second-best confidence
        Confidence margin
        Uncertainty status
        Complete probability distribution
    """

    if not isinstance(text, str):
        raise ValueError(
            "Input text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Input text must not be empty."
        )

    # --------------------------------------------------------
    # Tokenize input
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # --------------------------------------------------------
    # Generate model probabilities
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # --------------------------------------------------------
    # Rank predictions
    # --------------------------------------------------------

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True
    )

    primary_index = sorted_indices[0].item()
    second_index = sorted_indices[1].item()

    primary_emotion = model.config.id2label[
        primary_index
    ]

    second_emotion = model.config.id2label[
        second_index
    ]

    primary_confidence = sorted_probs[
        0
    ].item()

    second_confidence = sorted_probs[
        1
    ].item()

    # --------------------------------------------------------
    # Calculate confidence margin
    # --------------------------------------------------------

    confidence_margin = (
        primary_confidence
        - second_confidence
    )

    # --------------------------------------------------------
    # Determine uncertainty
    # --------------------------------------------------------

    uncertainty_status = determine_uncertainty(
        primary_confidence,
        confidence_margin
    )

    # --------------------------------------------------------
    # Complete probability distribution
    # --------------------------------------------------------

    probabilities_dict = {
        model.config.id2label[index]:
            round(probabilities[index].item(), 4)

        for index in range(
            len(probabilities)
        )
    }

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "text": text,

        "predicted_emotion":
            primary_emotion,

        "primary_emotion":
            primary_emotion,

        "confidence":
            primary_confidence,

        "second_best_emotion":
            second_emotion,

        "second_best_confidence":
            second_confidence,

        "confidence_margin":
            confidence_margin,

        "uncertainty_status":
            uncertainty_status,

        "probabilities":
            probabilities_dict
    }


# ============================================================
# TASK 4 VALIDATION
# ============================================================

if __name__ == "__main__":

    tokenizer, model = load_model()

    test_cases = [

        "I am very happy about my achievement.",

        "I feel sad and emotionally exhausted.",

        "I am extremely angry about the unfair workload.",

        "I am worried about my upcoming performance review.",

        "I received an unexpected promotion today.",

        "I am confused because my manager suddenly "
        "changed the deadline.",

        "I feel frustrated but also excited "
        "about the new opportunity.",

        "I don't know how I feel about the situation."
    ]

    print("=" * 75)
    print(
        "TASK 4 CONFIDENCE SCORE VALIDATION"
    )
    print("=" * 75)

    for number, text in enumerate(
        test_cases,
        start=1
    ):

        result = validate_confidence(
            text,
            tokenizer,
            model
        )

        print(
            f"\nTest {number}"
        )

        print(
            f"Text                  : "
            f"{result['text']}"
        )

        print(
            f"Predicted Emotion     : "
            f"{result['predicted_emotion']}"
        )

        print(
            f"Confidence            : "
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            f"Second Best           : "
            f"{result['second_best_emotion']} "
            f"({result['second_best_confidence'] * 100:.2f}%)"
        )

        print(
            f"Confidence Margin     : "
            f"{result['confidence_margin'] * 100:.2f}%"
        )

        print(
            f"Uncertainty Status    : "
            f"{result['uncertainty_status']}"
        )

        print(
            "Probabilities:"
        )

        for emotion, probability in (
            result["probabilities"].items()
        ):

            print(
                f"  {emotion:<10}: "
                f"{probability * 100:.2f}%"
            )

    # --------------------------------------------------------
    # Final validation summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "TASK 4 VALIDATION COMPLETE"
    )

    print("=" * 75)

    print(
        f"High confidence threshold     : "
        f"{HIGH_CONFIDENCE * 100:.0f}%"
    )

    print(
        f"Moderate confidence threshold : "
        f"{MODERATE_CONFIDENCE * 100:.0f}%"
    )

    print(
        f"Low confidence margin         : "
        f"{LOW_MARGIN * 100:.0f}%"
    )

    print(
        f"Very uncertain margin         : "
        f"{VERY_LOW_MARGIN * 100:.1f}%"
    )