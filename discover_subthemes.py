
from pathlib import Path
import argparse
import re

import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("outputs/clustered_reviews_kmeans.csv")
OUTPUT_DIR = Path("outputs")

DEFAULT_THEME = "free · subscription · pay · ads"
DEFAULT_N_SUBTHEMES = 4
DEFAULT_EXAMPLES = 5

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def make_slug(text):
    """Convert a theme name into a safe filename."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:60] or "theme"


def get_representative_reviews(reviews, embeddings, clustering, n_examples=5):
    """
    Select reviews closest to each K-Means cluster centre.

    These reviews represent the cluster in embedding space.
    They are not necessarily the most positive or negative reviews.
    """
    representative_parts = []

    for cluster_id in range(clustering.n_clusters):
        positions = [
            i for i, label in enumerate(reviews["subtheme_id"].tolist())
            if label == cluster_id
        ]

        if not positions:
            continue

        cluster_embeddings = embeddings[positions]
        centre = clustering.cluster_centers_[cluster_id]

        distances = (
            ((cluster_embeddings - centre) ** 2).sum(axis=1) ** 0.5
        )

        ranked_positions = [
            positions[i]
            for i in distances.argsort()[:n_examples]
        ]

        examples = reviews.iloc[ranked_positions].copy()
        examples["representative_rank"] = range(1, len(examples) + 1)

        representative_parts.append(examples)

    if not representative_parts:
        return pd.DataFrame()

    return pd.concat(representative_parts, ignore_index=True)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discover sub-themes within a SignalMap theme."
    )

    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help="Exact parent theme name from the clustered reviews CSV.",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_N_SUBTHEMES,
        help="Number of sub-themes to discover (default: 4).",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLES,
        help="Number of representative reviews per sub-theme.",
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output filename prefix.",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------

    if args.clusters < 2:
        raise ValueError("Choose at least 2 sub-themes.")

    if args.examples < 1:
        raise ValueError("Choose at least 1 representative review.")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("Loading clustered reviews...")
    df = pd.read_csv(INPUT_FILE)

    required = {"theme", "text_clean", "content", "score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing)}"
        )

    available_themes = sorted(df["theme"].dropna().unique())

    if args.theme not in available_themes:
        print("\nTheme not found. Available themes:")

        for theme in available_themes:
            print(f" - {theme}")

        raise ValueError(f"Unknown theme: {args.theme}")

    # --------------------------------------------------
    # FILTER SELECTED THEME
    # --------------------------------------------------

    reviews = df[df["theme"] == args.theme].copy()

    reviews["text_clean"] = (
        reviews["text_clean"].fillna("").astype(str).str.strip()
    )

    reviews = reviews[
        reviews["text_clean"].str.len() >= 8
    ].copy()

    if len(reviews) < args.clusters:
        raise ValueError(
            f"Only {len(reviews)} usable reviews are available. "
            f"Cannot create {args.clusters} clusters."
        )

    print(f"\nSelected theme: {args.theme}")
    print(f"Usable reviews: {len(reviews)}")
    print(f"Requested sub-themes: {args.clusters}")
    print(f"Representative reviews per sub-theme: {args.examples}")

    # --------------------------------------------------
    # GENERATE SENTENCE EMBEDDINGS
    # --------------------------------------------------

    print("\nLoading sentence transformer...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        reviews["text_clean"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # --------------------------------------------------
    # SUB-THEME CLUSTERING
    # --------------------------------------------------

    print("\nDiscovering sub-themes...")

    clustering = KMeans(
        n_clusters=args.clusters,
        random_state=42,
        n_init=10,
    )

    reviews["subtheme_id"] = clustering.fit_predict(embeddings)

    # --------------------------------------------------
    # ASSIGN NEUTRAL SUB-THEME LABELS
    # --------------------------------------------------

    labels = {
        cluster_id: f"Sub-theme {cluster_id + 1}"
        for cluster_id in range(args.clusters)
    }

    reviews["subtheme"] = reviews["subtheme_id"].map(labels)

    # --------------------------------------------------
    # GENERATE SUMMARY
    # --------------------------------------------------

    reviews["score"] = pd.to_numeric(
        reviews["score"], errors="coerce"
    )

    summary = (
        reviews.groupby("subtheme")
        .agg(
            review_count=("content", "size"),
            average_rating=("score", "mean"),
            negative_share_pct=(
                "score",
                lambda scores: (scores <= 2).mean() * 100,
            ),
        )
        .reset_index()
    )

    summary["average_rating"] = summary["average_rating"].round(2)

    summary["negative_share_pct"] = (
        summary["negative_share_pct"].round(1)
    )

    summary = summary.sort_values(
        "negative_share_pct",
        ascending=False,
    )

    # --------------------------------------------------
    # SELECT REPRESENTATIVE REVIEWS
    # --------------------------------------------------

    print("\nSelecting representative reviews...")

    representative_reviews = get_representative_reviews(
        reviews,
        embeddings,
        clustering,
        n_examples=args.examples,
    )

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or make_slug(args.theme)

    review_file = OUTPUT_DIR / f"{prefix}_subtheme_reviews.csv"
    summary_file = OUTPUT_DIR / f"{prefix}_subtheme_summary.csv"
    representative_file = (
        OUTPUT_DIR / f"{prefix}_representative_reviews.csv"
    )

    reviews.to_csv(review_file, index=False)
    summary.to_csv(summary_file, index=False)
    representative_reviews.to_csv(representative_file, index=False)

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("SIGNALMAP - SUB-THEME DISCOVERY RESULTS")
    print("=" * 65)

    print(f"\nParent theme: {args.theme}")
    print(f"Total reviews analysed: {len(reviews)}")
    print(f"Number of sub-themes: {args.clusters}")

    print("\nSUB-THEME SUMMARY\n")

    print(
        summary[
            [
                "subtheme",
                "review_count",
                "average_rating",
                "negative_share_pct",
            ]
        ].to_string(index=False)
    )

    print("\nREPRESENTATIVE REVIEWS\n")

    for subtheme, group in representative_reviews.groupby("subtheme"):
        print(f"\n{subtheme}")

        for _, review in group.iterrows():
            print(
                f"\n  Rating: {review['score']} | "
                f"App: {review['appName'] if 'appName' in review else 'Unknown'}"
            )
            print(f"  {review['content']}")

    print("\nOUTPUT FILES")
    print(f"Detailed reviews: {review_file}")
    print(f"Summary: {summary_file}")
    print(f"Representative reviews: {representative_file}")

    print("\nNote: Sub-theme labels require manual review.")
    print("Representative reviews are selected by embedding proximity.")
    print("Clustering groups similar text; it does not confirm root causes.")
    print("\nSub-theme discovery completed successfully!")


if __name__ == "__main__":
    main()
