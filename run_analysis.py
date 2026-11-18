from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_reviews, validate_reviews
from src.preprocessing import clean_reviews
from src.ml_pipeline import run_theme_discovery
from src.prioritization import build_priority_table

def main():
    parser = argparse.ArgumentParser(description="Run SignalMap's feedback theme discovery pipeline.")
    parser.add_argument("--input", default="data/raw/health_fitness_app_reviews.csv")
    parser.add_argument("--sample", type=int, default=5000, help="Maximum reviews to analyse")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument(
    "--clustering",
    choices=["kmeans", "agglomerative"],
    default="kmeans",
    help="Clustering algorithm to use",
)
    parser.add_argument("--tfidf-only", action="store_true", help="Skip transformer model and use TF-IDF baseline")
    args = parser.parse_args()

    df = load_reviews(args.input)
    print("DATASET VALIDATION")
    for key, value in validate_reviews(df).items():
        print(f"{key}: {value}")

    cleaned = clean_reviews(df)
    if len(cleaned) > args.sample:
        cleaned = cleaned.sample(n=args.sample, random_state=42).sort_index()
    print(f"\nAnalysing {len(cleaned):,} reviews...")
    clustered, metrics = run_theme_discovery(
    cleaned,
    n_clusters=args.clusters,
    use_transformer=not args.tfidf_only,
    clustering_method=args.clustering,
)
    priority = build_priority_table(clustered)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    clustered.to_csv(out / "clustered_reviews.csv", index=False)
    priority.to_csv(out / "priority_table.csv", index=False)
    print("\nMODEL METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved outputs to: {out.resolve()}")

if __name__ == "__main__":
    main()
