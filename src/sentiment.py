from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create VADER analyzer
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze sentiment using VADER.

    Returns:
        compound: overall sentiment score
        positive: positive score
        negative: negative score
        neutral: neutral score
        classification: Positive / Negative / Neutral
    """

    if text is None:
        raise ValueError("Text cannot be None.")

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    # VADER performs the actual sentiment analysis
    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]
    positive = scores["pos"]
    negative = scores["neg"]
    neutral = scores["neu"]

    # Classification based on compound score
    if compound >= 0.05:
        classification = "Positive"

    elif compound <= -0.05:
        classification = "Negative"

    else:
        classification = "Neutral"

    return {
        "text": text,
        "compound": compound,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "classification": classification
    }