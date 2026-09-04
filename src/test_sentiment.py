from sentiment import analyze_sentiment


def run_test(test_name, text, expected_classification):
    print(f"\n{test_name}")
    print("-" * 60)
    print("Input:", text)

    try:
        result = analyze_sentiment(text)

        print("Classification:", result["classification"])
        print("Compound:", result["compound"])
        print("Positive:", result["positive"])
        print("Negative:", result["negative"])
        print("Neutral:", result["neutral"])

        if result["classification"] == expected_classification:
            print("PASS")
        else:
            print(
                "FAIL - Expected:",
                expected_classification
            )

    except Exception as e:
        print("FAIL:", e)


print("========== TASK 3 VADER SENTIMENT TESTING ==========")


# Positive
run_test(
    "TEST 1 - Positive Sentiment",
    "I am extremely happy with my job!",
    "Positive"
)


# Negative
run_test(
    "TEST 2 - Negative Sentiment",
    "I am very unhappy and stressed with my workload.",
    "Negative"
)


# Neutral
run_test(
    "TEST 3 - Neutral Sentiment",
    "The meeting is scheduled for tomorrow.",
    "Neutral"
)


# Strong positive
run_test(
    "TEST 4 - Strong Positive",
    "I absolutely love working with my amazing team!",
    "Positive"
)


# Strong negative
run_test(
    "TEST 5 - Strong Negative",
    "I hate this job. The workload is terrible.",
    "Negative"
)


# Negation
run_test(
    "TEST 6 - Negation",
    "I am not happy with my current workload.",
    "Negative"
)


# Mixed sentiment
run_test(
    "TEST 7 - Mixed Sentiment",
    "The salary is good but the workload is stressful.",
    "Negative"
)


# Employee wellness example
run_test(
    "TEST 8 - Employee Wellness",
    "I feel supported by my manager and enjoy my work.",
    "Positive"
)


# Another employee example
run_test(
    "TEST 9 - Employee Wellness",
    "I feel exhausted and overwhelmed by my workload.",
    "Negative"
)


# Neutral employee example
run_test(
    "TEST 10 - Employee Wellness",
    "I completed my assigned task today.",
    "Neutral"
)