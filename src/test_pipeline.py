from pathlib import Path

from pipeline import (
    process_text,
    process_txt_file,
    process_csv_file,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


print("========== TASK 5 PIPELINE INTEGRATION TESTING ==========")


# TEST 1 - Direct text through complete pipeline
print("\nTEST 1 - Direct Text Pipeline")
print("-" * 60)

try:
    result = process_text(
        "I am very happy with my work!"
    )

    print("PASS")
    print("Original:", result["original_text"])
    print("Processed:", result["processed_text"])
    print("Sentiment:", result["sentiment"])
    print("Compound:", result["compound"])

except Exception as e:
    print("FAIL:", e)


# TEST 2 - Negative text
print("\nTEST 2 - Negative Text Pipeline")
print("-" * 60)

try:
    result = process_text(
        "I am stressed and unhappy with my workload."
    )

    print("PASS")
    print("Original:", result["original_text"])
    print("Processed:", result["processed_text"])
    print("Sentiment:", result["sentiment"])
    print("Compound:", result["compound"])

except Exception as e:
    print("FAIL:", e)


# TEST 3 - TXT file pipeline
print("\nTEST 3 - TXT File Pipeline")
print("-" * 60)

try:
    result = process_txt_file(
        DATA_DIR / "sample.txt"
    )

    print("PASS")
    print("Original:", result["original_text"])
    print("Processed:", result["processed_text"])
    print("Sentiment:", result["sentiment"])
    print("Compound:", result["compound"])

except Exception as e:
    print("FAIL:", e)


# TEST 4 - CSV file pipeline
print("\nTEST 4 - CSV File Pipeline")
print("-" * 60)

try:
    results = process_csv_file(
        DATA_DIR / "sample.csv"
    )

    print("PASS")
    print("Number of records:", len(results))

    for i, result in enumerate(results, start=1):
        print(f"\nRecord {i}")
        print("Text:", result["original_text"])
        print("Processed:", result["processed_text"])
        print("Sentiment:", result["sentiment"])
        print("Compound:", result["compound"])

except Exception as e:
    print("FAIL:", e)


# TEST 5 - Invalid input
print("\nTEST 5 - Invalid Input Pipeline")
print("-" * 60)

try:
    process_text("   ")
    print("FAIL - Invalid input was accepted.")

except Exception as e:
    print("PASS - Invalid input rejected correctly.")
    print("Validation:", e)


print("\n========== TASK 5 TESTING COMPLETED ==========")