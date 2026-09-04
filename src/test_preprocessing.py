from preprocessing import preprocess_text


def run_test(test_name, text, should_pass=True):
    print(f"\n{test_name}")
    print("-" * 50)

    try:
        result = preprocess_text(text)

        if should_pass:
            print("PASS")
            print("Original:", result["original_text"])
            print("Tokens:", result["tokens"])
            print("After stop-word handling:",
                  result["filtered_tokens"])
            print("Lemmatized:",
                  result["lemmatized_tokens"])
            print("Processed:",
                  result["processed_text"])
        else:
            print("FAIL - Invalid input was accepted.")

    except Exception as e:

        if should_pass:
            print("FAIL:", e)
        else:
            print("PASS - Invalid input rejected correctly.")
            print("Validation:", e)


print("========== TASK 2 PREPROCESSING TESTING ==========")


# 1. Normal text
run_test(
    "TEST 1 - Normal Text",
    "I am happy with my work."
)


# 2. Repeated spaces
run_test(
    "TEST 2 - Repeated Spaces",
    "I     am     very     happy     today."
)


# 3. Punctuation
run_test(
    "TEST 3 - Punctuation",
    "I am happy!!! My work is great."
)


# 4. Special characters
run_test(
    "TEST 4 - Special Characters",
    "I am happy @ work #today $100."
)


# 5. Negation
run_test(
    "TEST 5 - Negation",
    "I am not happy with my workload."
)


# 6. Important sentiment words
run_test(
    "TEST 6 - Sentiment Words",
    "I am very happy but never satisfied."
)


# 7. Lemmatization
run_test(
    "TEST 7 - Lemmatization",
    "Employees are working on different projects."
)


# 8. URL noise
run_test(
    "TEST 8 - URL Noise",
    "I am happy. Visit https://example.com for details."
)


# 9. Empty text
run_test(
    "TEST 9 - Empty Text",
    "   ",
    should_pass=False
)


# 10. None input
run_test(
    "TEST 10 - None Input",
    None,
    should_pass=False
)