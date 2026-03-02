import argparse
import os
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)


def main():
    parser = argparse.ArgumentParser(description="Extract sentiment features using VADER")
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

    sia = SentimentIntensityAnalyzer()

    # Apply VADER sentiment analysis
    sentiments = df[args.text_col].fillna("").apply(sia.polarity_scores)
    df["sentiment_pos"] = sentiments.apply(lambda x: x["pos"])
    df["sentiment_neg"] = sentiments.apply(lambda x: x["neg"])
    df["sentiment_neu"] = sentiments.apply(lambda x: x["neu"])
    df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])

    print(f"Mean compound sentiment: {df['sentiment_compound'].mean():.3f}")

    os.makedirs(args.output, exist_ok=True)
    df.to_parquet(os.path.join(args.output, "data.parquet"), index=False)
    print(f"Wrote {len(df)} rows to output")


if __name__ == "__main__":
    main()
