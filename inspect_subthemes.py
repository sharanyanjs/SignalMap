import pandas as pd

df = pd.read_csv("outputs/subtheme_reviews.csv")

for name, group in df.groupby("subtheme"):
    print("\n" + "=" * 70)
    print(name, "| REVIEWS:", len(group))
    print("=" * 70)

    examples = group.sort_values("score").head(8)

    for _, row in examples.iterrows():
        print(f"\nRating: {row['score']}")
        print(row["content"])