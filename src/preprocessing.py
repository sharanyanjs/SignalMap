import re
import pandas as pd

def clean_reviews(df: pd.DataFrame, min_chars: int = 8) -> pd.DataFrame:
    out = df.copy()
    out["text_clean"] = (
        out["content"].fillna("").astype(str)
        .str.replace(r"https?://\S+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out = out[out["text_clean"].str.len() >= min_chars].copy()
    out = out.drop_duplicates(subset=["appName", "text_clean"])
    out["review_date"] = pd.to_datetime(out["at"], errors="coerce", utc=True)
    out["rating_sentiment"] = out["score"].map(
        lambda x: "positive" if x >= 4 else ("neutral" if x == 3 else "negative")
    )
    return out.reset_index(drop=True)
