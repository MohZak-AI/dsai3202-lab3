import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Generate SBERT semantic embeddings")
    parser.add_argument("--data", type=str, required=True, help="Input uri_folder")
    parser.add_argument("--output", type=str, required=True, help="Output uri_folder")
    parser.add_argument("--text_col", type=str, default="reviewText")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Read parquet files
    input_files = [f for f in os.listdir(args.data) if f.endswith(".parquet")]
    df = pd.concat(
        [pd.read_parquet(os.path.join(args.data, f)) for f in input_files],
        ignore_index=True,
    )
    print(f"Loaded {len(df)} rows")

    # Load sentence-transformers model
    model = SentenceTransformer(args.model_name)

    texts = df[args.text_col].fillna("").tolist()

    # Encode in batches
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")

    # Add embedding columns
    embedding_dim = embeddings.shape[1]
    for i in range(embedding_dim):
        df[f"bert_embedding_{i}"] = embeddings[:, i]

    os.makedirs(args.output, exist_ok=True)
    df.to_parquet(os.path.join(args.output, "data.parquet"), index=False)
    print(f"Wrote {len(df)} rows with {embedding_dim} embedding dims to output")


if __name__ == "__main__":
    main()
