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
# MULTI-LABEL DECISION PARAMETERS
# ============================================================
#
# IMPORTANT:
# The current model is trained as a single-label classifier
# using softmax probabilities.
#
# These thresholds provide a heuristic interpretation of
# multiple strong emotions. This is NOT a true multi-label
# classifier trained with sigmoid + multi-hot labels.
# ============================================================

MULTI_LABEL_THRESHOLD = 0.17
RELATIVE_TO_PRIMARY = 0.90


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path=MODEL_PATH):
    """
    Load tokenizer and trained emotion model.
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
# MULTI-LABEL EMOTION ANALYSIS
# ============================================================

def analyze_multi_label(
    text,
    model_path=MODEL_PATH
):
    """
    Analyze primary and potentially secondary emotions.

    The model produces a single-label softmax distribution.
    Secondary emotions are identified using configurable
    probability thresholds.

    Returns:
        Primary emotion
        Primary confidence
        Secondary emotions
        Detected emotions
        Emotion count
        Analysis type
        Complete probability distribution
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(text, str):
        raise ValueError(
            "Text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Text cannot be empty."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Tokenize input
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # --------------------------------------------------------
    # Generate probabilities
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
    # Map probabilities to emotion labels
    # --------------------------------------------------------

    id2label = model.config.id2label

    probability_map = {
        id2label[index]:
            probability.item()

        for index, probability in enumerate(
            probabilities
        )
    }

    # --------------------------------------------------------
    # Rank emotions
    # --------------------------------------------------------

    ranked_emotions = sorted(
        probability_map.items(),
        key=lambda item: item[1],
        reverse=True
    )

    primary_emotion = (
        ranked_emotions[0][0]
    )

    primary_probability = (
        ranked_emotions[0][1]
    )

    # --------------------------------------------------------
    # Identify secondary emotions
    # --------------------------------------------------------

    secondary_emotions = []

    for emotion, probability in (
        ranked_emotions[1:]
    ):

        meets_absolute_threshold = (
            probability >= MULTI_LABEL_THRESHOLD
        )

        meets_relative_threshold = (
            probability
            >= primary_probability
            * RELATIVE_TO_PRIMARY
        )

        if (
            meets_absolute_threshold
            and meets_relative_threshold
        ):
            secondary_emotions.append(
                emotion
            )

    # --------------------------------------------------------
    # Combine detected emotions
    # --------------------------------------------------------

    detected_emotions = [
        primary_emotion
    ] + secondary_emotions

    # --------------------------------------------------------
    # Determine analysis type
    # --------------------------------------------------------

    emotion_count = len(
        detected_emotions
    )

    if emotion_count == 1:

        analysis_type = (
            "Single emotion"
        )

    elif emotion_count == 2:

        analysis_type = (
            "Two-emotion combination"
        )

    else:

        analysis_type = (
            "Mixed emotion"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "text": text,

        "primary_emotion":
            primary_emotion,

        "primary_confidence":
            round(
                primary_probability,
                4
            ),

        "secondary_emotions":
            secondary_emotions,

        "detected_emotions":
            detected_emotions,

        "emotion_count":
            emotion_count,

        "analysis_type":
            analysis_type,

        "threshold":
            MULTI_LABEL_THRESHOLD,

        "relative_threshold":
            RELATIVE_TO_PRIMARY,

        "probabilities": {
            emotion: round(
                probability,
                4
            )
            for emotion, probability
            in ranked_emotions
        }
    }


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_analysis(result):
    """
    Display a multi-label emotion analysis result.
    """

    print(
        "\n" + "=" * 75
    )

    print(
        "MULTI-LABEL EMOTION ANALYSIS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nText:\n{result['text']}"
    )

    print(
        f"\nPrimary Emotion: "
        f"{result['primary_emotion']}"
    )

    print(
        f"Primary Confidence: "
        f"{result['primary_confidence']:.2%}"
    )

    print(
        f"\nSecondary Emotions: "
        f"{result['secondary_emotions']}"
    )

    print(
        f"Detected Emotion Count: "
        f"{result['emotion_count']}"
    )

    print(
        f"Analysis Type: "
        f"{result['analysis_type']}"
    )

    print(
        "\nEmotion Probabilities:"
    )

    for emotion, probability in (
        result["probabilities"].items()
    ):

        print(
            f"{emotion:<10} "
            f"{probability:.2%}"
        )

    print(
        f"\nAbsolute Threshold: "
        f"{result['threshold']:.2%}"
    )

    print(
        f"Relative Threshold: "
        f"{result['relative_threshold']:.0%}"
    )


# ============================================================
# TASK 3 VALIDATION
# ============================================================

if __name__ == "__main__":

    test_cases = [

        "I feel extremely happy about my achievement.",

        "I feel sad and emotionally exhausted.",

        "I am extremely frustrated by this unfair workload.",

        "I am frustrated with the workload but excited "
        "about the new opportunity.",

        "I am happy about my promotion but nervous "
        "about the new responsibilities.",

        "I feel anxious, frustrated, and unsupported "
        "by my team."
    ]

    print(
        "\n" + "=" * 75
    )

    print(
        "TASK 3 MULTI-LABEL VALIDATION SUMMARY"
    )

    print(
        "=" * 75
    )

    for number, text in enumerate(
        test_cases,
        start=1
    ):

        result = analyze_multi_label(
            text
        )

        print(
            f"\nTest {number}"
        )

        print(
            f"Text      : "
            f"{text}"
        )

        print(
            f"Primary   : "
            f"{result['primary_emotion']}"
        )

        print(
            f"Confidence: "
            f"{result['primary_confidence']:.2%}"
        )

        print(
            f"Secondary : "
            f"{result['secondary_emotions']}"
        )

        print(
            f"Count     : "
            f"{result['emotion_count']}"
        )

        print(
            f"Type      : "
            f"{result['analysis_type']}"
        )

    print(
        "\n" + "=" * 75
    )

    print(
        "TASK 3 VALIDATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Absolute threshold: "
        f"{MULTI_LABEL_THRESHOLD:.0%}"
    )

    print(
        f"Relative threshold: "
        f"{RELATIVE_TO_PRIMARY:.0%}"
    )