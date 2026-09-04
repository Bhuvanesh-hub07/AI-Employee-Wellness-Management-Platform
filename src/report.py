import pandas as pd

from preprocessing import preprocess_text
from sentiment import analyze_sentiment


def generate_sentiment_report(csv_path):
    """
    Generate an initial sentiment report from a CSV file.
    """

    # Read input data
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    results = []

    for index, row in df.iterrows():

        original_text = row["text"]

        # Skip empty values
        if pd.isna(original_text):
            continue

        original_text = str(original_text).strip()

        if not original_text:
            continue

        # Preprocessing
        preprocessing_result = preprocess_text(original_text)

        # VADER sentiment analysis
        sentiment_result = analyze_sentiment(original_text)

        # Combine results
        results.append({
            "sample_number": len(results) + 1,
            "original_text": original_text,
            "processed_text": preprocessing_result["processed_text"],
            "sentiment": sentiment_result["classification"],
            "compound": sentiment_result["compound"],
            "positive": sentiment_result["positive"],
            "negative": sentiment_result["negative"],
            "neutral": sentiment_result["neutral"]
        })

    if not results:
        raise ValueError("No valid text samples found.")

    return pd.DataFrame(results)


def save_report(report_df, output_path):
    """Save report as CSV."""

    report_df.to_csv(
        output_path,
        index=False
    )

    print(f"Report saved to: {output_path}")