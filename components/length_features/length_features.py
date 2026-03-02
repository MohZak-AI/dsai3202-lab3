import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Compute review length features")
    parser.add_argument("--data", type=str, required=True, help="Input uri_folder")
    parser.add_argument("--output", type=str, required=True, help="Output uri_folder")
    parser.add_argument("--text_col", type=str, default="reviewText")
    args = parser.parse_args()

    # Read parquet files
    input_files = [f for f in os.listdir(args.data) if f.endswith(".parquet")]
    df = pd.concat(
        [pd.read_parquet(os.path.join(args.data, f)) for f in input_files],
        ignore_index=True,
    )
    print(f"Loaded {len(df)} rows")

    # Compute length features
    df["review_length_words"] = df[args.text_col].fillna("").apply(lambda x: len(x.split()))
    df["review_length_chars"] = df[args.text_col].fillna("").apply(len)

    print(f"Mean word length: {df['review_length_words'].mean():.1f}")
    print(f"Mean char length: {df['review_length_chars'].mean():.1f}")

    os.makedirs(args.output, exist_ok=True)
    df.to_parquet(os.path.join(args.output, "data.parquet"), index=False)
    print(f"Wrote {len(df)} rows to output")


if __name__ == "__main__":
    main()
