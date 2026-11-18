
import pandas as pd
from src.preprocessing import clean_reviews
from src.prioritization import build_priority_table


def test_clean_and_prioritise():
    df = pd.DataFrame({
        "content": [
            "App crashes during login every time",
            "Great app",
            "Payment fails at checkout"
        ],
        "score": [1, 5, 2],
        "at": pd.to_datetime(
            ["2025-01-01", "2025-01-02", "2025-01-03"],
            utc=True
        ),
        "appName": ["A", "A", "B"],
        "thumbsUpCount": [0, 1, 0],
        "tag": ["fitness", "fitness", "fitness"],
    })

    cleaned = clean_reviews(df)

    # All three reviews meet the minimum text length.
    assert len(cleaned) == 3

    cleaned["theme"] = [
        "login crash",
        "general praise",
        "payment issue"
    ]

    table = build_priority_table(cleaned)

    # Each theme should have one priority row.
    assert len(table) == 3
    assert "priority_score" in table.columns
