from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"content", "score", "at", "appName"}

def load_reviews(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["at"] = pd.to_datetime(df["at"], errors="coerce", utc=True)
    df["appName"] = df["appName"].fillna("Unknown").astype(str)
    if "tag" not in df:
        df["tag"] = "Unspecified"
    if "thumbsUpCount" not in df:
        df["thumbsUpCount"] = 0
    df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0)
    return df

def validate_reviews(df: pd.DataFrame) -> dict:
    usable = df[df["content"].str.len() >= 8]
    return {
        "rows": int(len(df)),
        "usable_text_rows": int(len(usable)),
        "missing_text": int((df["content"].str.len() == 0).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_review_text": int(df.loc[df["content"].str.len() > 0, "content"].duplicated().sum()),
        "apps": int(df["appName"].nunique()),
        "date_min": df["at"].min(),
        "date_max": df["at"].max(),
    }
