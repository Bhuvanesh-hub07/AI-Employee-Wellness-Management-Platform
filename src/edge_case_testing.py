"""
Task 8 - Edge Case Validation.

Tests the integrated employee wellness pipeline against:
- Positive feedback
- Negative feedback
- Neutral feedback
- Mixed emotions
- Very short text
- Long text
- Informal text
- Emoji input
- Ambiguous text
- Empty input
- Whitespace input
- None input
- Numeric input

The test suite verifies that normal inputs are processed
successfully and invalid inputs are handled safely.
"""

from pipeline import process_text


# ---------------------------------------------------------------------
# Edge-case test data
# ---------------------------------------------------------------------

EDGE_CASES = [
    (
        "Positive feedback",
        "I am extremely happy and proud of my work.",
    ),
    (
        "Negative feedback",
        "I am frustrated and unhappy with my workload.",
    ),
    (
        "Neutral feedback",
        "The team meeting is scheduled for tomorrow.",
    ),
    (
        "Mixed emotions",
        "I am nervous about the presentation but excited "
        "to show my work.",
    ),
    (
        "Very short input",
        "Happy",
    ),
    (
        "Long input",
        "I have been working on this project for several weeks "
        "and although I have learned many new skills and received "
        "support from my teammates, I sometimes feel stressed "
        "because of deadlines, changing requirements, meetings, "
        "and the pressure to complete everything correctly. "
        "At the same time, I feel proud of the progress I have "
        "made and excited about the opportunity to improve my "
        "technical and communication skills.",
    ),
    (
        "Informal text",
        "Honestly, work has been kinda stressful lately lol",
    ),
    (
        "Emoji input",
        "I finally completed my project! 😊🎉",
    ),
    (
        "Ambiguous input",
        "I guess things are okay.",
    ),
    (
        "Empty input",
        "",
    ),
    (
        "Whitespace input",
        "     ",
    ),
    (
        "Invalid input - None",
        None,
    ),
    (
        "Invalid input - number",
        12345,
    ),
]


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def is_expected_invalid_input(test_input):
    """
    Determine whether an input is expected to be rejected.

    Invalid inputs include:
    - None
    - Empty strings
    - Whitespace-only strings
    - Non-string values
    """

    if test_input is None:
        return True

    if not isinstance(test_input, str):
        return True

    return not test_input.strip()


def display_result(result):
    """Display the important pipeline output fields."""

    print(f"Sentiment       : {result['sentiment']}")
    print(f"Emotion         : {result['emotion']}")
    print(f"Confidence      : {result['confidence']:.2%}")
    print(
        f"Confidence Status: "
        f"{result['confidence_status']}"
    )
    print(
        f"Detected Emotions: "
        f"{', '.join(result['detected_emotions'])}"
    )
    print(
        f"Analysis Type   : "
        f"{result['analysis_type']}"
    )
    print(
        f"Emotion Count   : "
        f"{result['emotion_count']}"
    )


def run_edge_case_test(
    test_number,
    description,
    test_input,
):
    """
    Run one edge-case test and return its status.

    Returns:
        Tuple containing:
        - True/False for a normal successful test
        - True/False for an expected handled error
    """

    print("\n" + "=" * 80)
    print(
        f"TEST {test_number}: {description}"
    )
    print("=" * 80)

    print(
        f"Input: {repr(test_input)}"
    )

    try:
        result = process_text(
            test_input
        )

        print("\nRESULT: PASS")
        display_result(result)

        return True, False

    except Exception as error:

        print("\nRESULT: HANDLED ERROR")
        print(
            f"Error: {error}"
        )

        if is_expected_invalid_input(
            test_input
        ):
            return False, True

        print(
            "Unexpected failure detected."
        )

        return False, False


# ---------------------------------------------------------------------
# Main test execution
# ---------------------------------------------------------------------

def main():
    """Run all Task 8 edge-case tests."""

    print("\n")
    print("#" * 80)
    print("TASK 8 - EDGE CASE VALIDATION")
    print("#" * 80)

    total_tests = len(
        EDGE_CASES
    )

    passed_tests = 0
    handled_errors = 0
    unexpected_failures = 0

    for number, (
        description,
        test_input,
    ) in enumerate(
        EDGE_CASES,
        start=1,
    ):

        print("\n" + "-" * 80)
        print(
            f"Running Test {number}/{total_tests}: "
            f"{description}"
        )

        passed, handled_error = (
            run_edge_case_test(
                number,
                description,
                test_input,
            )
        )

        if passed:
            passed_tests += 1

        elif handled_error:
            handled_errors += 1

        else:
            unexpected_failures += 1

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print("\n")
    print("#" * 80)
    print("TASK 8 - EDGE CASE TEST SUMMARY")
    print("#" * 80)

    print(
        f"Total Tests             : "
        f"{total_tests}"
    )

    print(
        f"Normal Inputs Passed    : "
        f"{passed_tests}"
    )

    print(
        f"Invalid Inputs Handled  : "
        f"{handled_errors}"
    )

    print(
        f"Unexpected Failures     : "
        f"{unexpected_failures}"
    )

    if unexpected_failures == 0:
        print(
            "\nEdge-case testing completed successfully."
        )
    else:
        print(
            "\nEdge-case testing completed "
            "with unexpected failures."
        )

    print("#" * 80)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()