from ingestion import ingest_direct_text, ingest_txt, ingest_csv
from preprocessing import preprocess_text
from sentiment import analyze_sentiment


def process_text(text):
    """
    Complete pipeline for one text input.

    Flow:
    Input → Validation → Preprocessing → VADER
    """

    # Step 1: Validate input
    valid_text = ingest_direct_text(text)

    # Step 2: Preprocess
    preprocessing_result = preprocess_text(valid_text)

    # Step 3: Sentiment analysis
    sentiment_result = analyze_sentiment(valid_text)

    # Combine results
    return {
        "original_text": valid_text,
        "processed_text": preprocessing_result["processed_text"],
        "sentiment": sentiment_result["classification"],
        "compound": sentiment_result["compound"],
        "positive": sentiment_result["positive"],
        "negative": sentiment_result["negative"],
        "neutral": sentiment_result["neutral"]
    }


def process_txt_file(file_path):
    """
    Complete pipeline for a TXT file.
    """

    text = ingest_txt(file_path)

    return process_text(text)


def process_csv_file(file_path):
    """
    Complete pipeline for a CSV file.
    """

    texts = ingest_csv(file_path)

    results = []

    for text in texts:
        results.append(process_text(text))

    return results