import argparse
import os
import pandas as pd


def load_parquet_folder(folder):
    """Load all parquet files from a folder into a single DataFrame."""
    files = [f for f in os.listdir(folder) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")
    return pd.concat(
        [pd.read_parquet(os.path.join(folder, f)) for f in files],
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Merge all feature outputs")
    parser.add_argument("--length", type=str, required=True, help="Length features folder")
    parser.add_argument("--sentiment", type=str, required=True, help="Sentiment features folder")
    parser.add_argument("--tfidf", type=str, required=True, help="TF-IDF features folder")
    parser.add_argument("--sbert", type=str, required=True, help="SBERT embeddings folder")
    parser.add_argument("--output", type=str, required=True, help="Output uri_folder")
    parser.add_argument("--join_keys", type=str, default="asin,reviewerID")
    args = parser.parse_args()

    join_keys = args.join_keys.split(",")

    length_df = load_parquet_folder(args.length)
    sentiment_df = load_parquet_folder(args.sentiment)
    tfidf_df = load_parquet_folder(args.tfidf)
    sbert_df = load_parquet_folder(args.sbert)

    print(f"Length: {length_df.shape}, Sentiment: {sentiment_df.shape}")
    print(f"TF-IDF: {tfidf_df.shape}, SBERT: {sbert_df.shape}")

    # Get feature-only columns from each source (exclude base columns already in length_df)
    base_cols = set(length_df.columns)

    sentiment_feature_cols = [c for c in sentiment_df.columns if c not in base_cols]
    tfidf_feature_cols = [c for c in tfidf_df.columns if c not in base_cols]
    sbert_feature_cols = [c for c in sbert_df.columns if c not in base_cols]

    # Start with length features (has all base columns + length features)
    merged = length_df.copy()

    # Merge sentiment features
    merged = merged.merge(
        sentiment_df[join_keys + sentiment_feature_cols],
        on=join_keys,
        how="left",
    )

    # Merge TF-IDF features
    merged = merged.merge(
        tfidf_df[join_keys + tfidf_feature_cols],
        on=join_keys,
        how="left",
    )

    # Merge SBERT features
    merged = merged.merge(
        sbert_df[join_keys + sbert_feature_cols],
        on=join_keys,
        how="left",
    )

    print(f"Merged shape: {merged.shape}")

    os.makedirs(args.output, exist_ok=True)
    merged.to_parquet(os.path.join(args.output, "features.parquet"), index=False)
    print(f"Wrote {len(merged)} rows with {len(merged.columns)} columns")


if __name__ == "__main__":
    main()
