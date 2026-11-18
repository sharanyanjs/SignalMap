
import pandas as pd


def weekly_theme_trends(clustered: pd.DataFrame) -> pd.DataFrame:
    data = clustered.dropna(subset=["review_date"]).copy()

    if data.empty:
        return pd.DataFrame(columns=["week", "theme", "reviews"])

    # Remove timezone information before converting dates to weekly periods.
    # This avoids pandas' timezone conversion warning.
    if data["review_date"].dt.tz is not None:
        data["review_date"] = data["review_date"].dt.tz_localize(None)

    data["week"] = (
        data["review_date"]
        .dt.to_period("W")
        .dt.start_time
    )

    return (
        data.groupby(["week", "theme"])
        .size()
        .reset_index(name="reviews")
        .sort_values("week")
    )


def representative_reviews(
    clustered: pd.DataFrame,
    theme: str,
    limit: int = 5
) -> pd.DataFrame:

    cols = [
        "appName",
        "score",
        "review_date",
        "content",
        "thumbsUpCount"
    ]

    return (
        clustered[clustered["theme"] == theme]
        .sort_values(
            ["score", "thumbsUpCount"],
            ascending=[True, False]
        )[cols]
        .head(limit)
    )
