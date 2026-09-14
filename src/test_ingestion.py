from pathlib import Path

from ingestion import (
    ingest_direct_text,
    ingest_txt,
    ingest_csv,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

print("========== TASK 1 TESTING ==========")


# Test 1: Direct text
print("\nTEST 1 - Direct Text")

try:
    result = ingest_direct_text(
        "I am happy with my work today."
    )

    print("PASS")
    print("Result:", result)

except Exception as e:
    print("FAIL:", e)


# Test 2: Empty text
print("\nTEST 2 - Empty Text")

try:
    ingest_direct_text("   ")
    print("FAIL - Empty text was accepted.")

except Exception as e:
    print("PASS")
    print("Validation:", e)


# Test 3: TXT file
print("\nTEST 3 - TXT File")

try:
    result = ingest_txt(DATA_DIR / "sample.txt")

    print("PASS")
    print("Result:", result)

except Exception as e:
    print("FAIL:", e)


# Test 4: CSV file
print("\nTEST 4 - CSV File")

try:
    result = ingest_csv(DATA_DIR / "sample.csv")

    print("PASS")
    print("Number of texts:", len(result))
    print("Texts:", result)

except Exception as e:
    print("FAIL:", e)


# Test 5: None input
print("\nTEST 5 - None Input")

try:
    ingest_direct_text(None)
    print("FAIL - None input was accepted.")

except Exception as e:
    print("PASS")
    print("Validation:", e)


# Test 6: Non-string input
print("\nTEST 6 - Non-String Input")

try:
    ingest_direct_text(12345)
    print("FAIL - Non-string input was accepted.")

except Exception as e:
    print("PASS")
    print("Validation:", e)


# Test 7: Missing TXT file
print("\nTEST 7 - Missing TXT File")

try:
    ingest_txt(DATA_DIR / "missing.txt")
    print("FAIL - Missing file was accepted.")

except Exception as e:
    print("PASS")
    print("Validation:", e)


# Test 8: Unsupported file type
print("\nTEST 8 - Unsupported File Type")

try:
    ingest_txt(DATA_DIR / "sample.csv")
    print("FAIL - Unsupported file type was accepted.")

except Exception as e:
    print("PASS")
    print("Validation:", e)


# Test 9: CSV without text column
print("\nTEST 9 - Invalid CSV Format")

try:
    ingest_csv(DATA_DIR / "invalid.csv")
    print("FAIL - Invalid CSV was accepted.")

except Exception as e:
    print("PASS")
    print("Validation:", e)