
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_REVIEWS = Path("outputs/subtheme_reviews.csv")
INPUT_SUMMARY = Path("outputs/subtheme_summary.csv")
OUTPUT_REPORT = Path("outputs/product_insights.md")

INVESTIGATION_AREAS = {
    "Excessive advertising": (
        "Advertising frequency, interruption, and the impact "
        "of advertising on the free-user experience."
    ),
    "Paywall and free-tier dissatisfaction": (
        "Which features belong in the free tier and whether "
        "premium restrictions are affecting customer retention."
    ),
    "Subscription and billing problems": (
        "Subscription activation, payment transparency, "
        "cancellation, refunds, and premium access."
    ),
    "Privacy and product experience": (
        "Data-sharing consent, privacy controls, account "
        "experience, and disruptive interface changes."
    ),
}


# --------------------------------------------------
# GENERATE REPORT
# --------------------------------------------------

def main():

    print("Loading sub-theme data...")

    reviews = pd.read_csv(INPUT_REVIEWS)
    summary = pd.read_csv(INPUT_SUMMARY)

    required_columns = {
        "subtheme", "score", "content"
    }

    missing = required_columns - set(reviews.columns)

    if missing:
        raise ValueError(
            f"Missing review columns: {sorted(missing)}"
        )

    report = []

    report.append("# SignalMap - Product Insights Report")
    report.append("")
    report.append(
        "Evidence-based exploration of customer feedback."
    )
    report.append("")
    report.append(
        f"Total reviews analysed: **{len(reviews)}**"
    )
    report.append("")
    report.append("---")
    report.append("")

    for _, row in summary.iterrows():

        theme = row["subtheme"]

        theme_reviews = reviews[
            reviews["subtheme"] == theme
        ].copy()

        report.append(f"## {theme}")
        report.append("")

        report.append(
            f"- Review count: **{int(row['review_count'])}**"
        )
        report.append(
            f"- Average rating: **{row['average_rating']} / 5**"
        )
        report.append(
            f"- Negative reviews (ratings 1-2): "
            f"**{row['negative_share_pct']}%**"
        )

        report.append("")

        report.append("### Customer evidence")
        report.append("")

        # Select up to three informative low-rated reviews
        examples = theme_reviews[
            theme_reviews["score"] <= 2
        ].copy()

        examples["content"] = (
            examples["content"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        examples = examples[
            examples["content"].str.len() >= 15
        ]

        examples = examples.sort_values(
            "content",
            key=lambda values: values.str.len(),
            ascending=False,
        ).head(3)

        if examples.empty:
            report.append(
                "No suitable low-rated examples found."
            )
        else:
            for _, example in examples.iterrows():

                quote = (
                    example["content"]
                    .replace("\n", " ")
                    .replace("\r", " ")
                )

                report.append(f"> {quote}")
                report.append("")

        report.append("### Suggested PM investigation")
        report.append("")

        recommendation = INVESTIGATION_AREAS.get(
            theme,
            "Review the customer feedback and investigate "
            "the underlying product experience."
        )

        report.append(recommendation)
        report.append("")

        report.append(
            "**Validation note:** This category is based on "
            "machine-learning clustering and requires human "
            "review. The suggested investigation is a "
            "hypothesis, not a confirmed root cause."
        )

        report.append("")
        report.append("---")
        report.append("")

    # Save report
    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nProduct insights report generated!")
    print(f"Report location: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
