import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    parser = argparse.ArgumentParser(description="Compute TF-IDF features")
    parser.add_argument("--train", type=str, required=True, help="Train split uri_folder")
    parser.add_argument("--val", type=str, required=True, help="Val split uri_folder")
    parser.add_argument("--test", type=str, required=True, help="Test split uri_folder")
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="reviewText")
    parser.add_argument("--max_features", type=int, default=500)
    args = parser.parse_args()

    def load_parquet(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".parquet")]
        return pd.concat(
            [pd.read_parquet(os.path.join(folder, f)) for f in files],
            ignore_index=True,
        )

    train_df = load_parquet(args.train)
    val_df = load_parquet(args.val)
    test_df = load_parquet(args.test)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Fit TF-IDF on training data only (avoid data leakage)
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2),
    )

    train_text = train_df[args.text_col].fillna("")
    val_text = val_df[args.text_col].fillna("")
    test_text = test_df[args.text_col].fillna("")

    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)

    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]

    # Add TF-IDF columns to dataframes
    for df, matrix in [(train_df, train_tfidf), (val_df, val_tfidf), (test_df, test_tfidf)]:
        tfidf_arr = matrix.toarray()
        for i, col_name in enumerate(feature_names):
            df[col_name] = tfidf_arr[:, i]

    # Write outputs
    for df, out_path, label in [
        (train_df, args.train_out, "train"),
        (val_df, args.val_out, "val"),
        (test_df, args.test_out, "test"),
    ]:
        os.makedirs(out_path, exist_ok=True)
        df.to_parquet(os.path.join(out_path, "data.parquet"), index=False)
        print(f"Wrote {label}: {len(df)} rows, {len(feature_names)} TF-IDF features")


if __name__ == "__main__":
    main()
