import argparse
import os
import re
import string
import pandas as pd


def normalize_text(text: str) -> str:
    """Lowercase, remove URLs, numbers, punctuation, and trim whitespace."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Replace URLs with empty string
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Replace numbers with empty string
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Trim whitespace (collapse multiple spaces and strip)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Normalize review text")
    parser.add_argument("--data", type=str, required=True, help="Input uri_folder path")
    parser.add_argument("--output", type=str, required=True, help="Output uri_folder path")
    parser.add_argument("--text_col", type=str, default="reviewText", help="Column to normalize")
    args = parser.parse_args()

    # Read all parquet files from input folder
    input_files = [f for f in os.listdir(args.data) if f.endswith(".parquet")]
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {args.data}")

    df = pd.concat(
        [pd.read_parquet(os.path.join(args.data, f)) for f in input_files],
        ignore_index=True,
    )

    print(f"Loaded {len(df)} rows")

    # Normalize text column
    df[args.text_col] = df[args.text_col].apply(normalize_text)

    # Filter out empty or very short reviews (< 10 characters)
    before = len(df)
    df = df[df[args.text_col].str.len() >= 10].reset_index(drop=True)
    print(f"Filtered {before - len(df)} short/empty reviews, {len(df)} remaining")

    # Write output
    os.makedirs(args.output, exist_ok=True)
    df.to_parquet(os.path.join(args.output, "normalized.parquet"), index=False)
    print(f"Wrote {len(df)} rows to {args.output}/normalized.parquet")


if __name__ == "__main__":
    main()