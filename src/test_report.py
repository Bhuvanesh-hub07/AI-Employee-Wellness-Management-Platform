from pathlib import Path

from report import generate_sentiment_report, save_report


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

print("========== TASK 4 REPORT TESTING ==========")

input_file = DATA_DIR / "sample.csv"
output_file = OUTPUT_DIR / "milestone1_sentiment_report.csv"


try:
    report = generate_sentiment_report(input_file)

    print("\nREPORT GENERATED SUCCESSFULLY")
    print("-" * 70)

    print("Number of analyzed samples:", len(report))

    print("\nReport:")
    print(report.to_string(index=False))

    save_report(report, output_file)

    print("\nPASS - Task 4 completed successfully.")

except Exception as e:
    print("\nFAIL:", e)