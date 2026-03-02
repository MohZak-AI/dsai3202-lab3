# DSAI3202 Lab 3 & Lab 4

This repo covers Labs 3 and 4 for Assignment 1. Lab 3 was about cleaning and preparing the data in Databricks. Lab 4 picks up from there and builds a full feature engineering pipeline on Azure ML.

## Lab 3 - Data Preprocessing in Azure

In this lab I loaded the raw Amazon Electronics reviews into Spark, cleaned them up, joined them with product metadata, and saved everything as a Gold dataset in Delta format. This gave me a solid base to work with in Lab 4.

### Notebooks

**01_load_and_clean_reviews.ipynb** - Reads the raw data, drops nulls and duplicates, fixes data types, trims strings. Basically just getting the data into a usable state.

**02_enrich_with_metadata.ipynb** - Takes the cleaned reviews and joins them with product metadata on `asin`. This adds things like product category and brand info that are useful for modeling later.

**03_write_gold_features_v1.ipynb** - Applies some final transformations and writes the output as Delta. This is the "gold" dataset that gets sampled and registered as a data asset for Lab 4.

---

## Lab 4 - Feature Engineering Pipeline

This is where the bulk of the work happened. I built an Azure ML pipeline that takes the sampled gold dataset and runs it through several feature extraction steps, then merges everything into one big feature table.

### Setup

Before building the pipeline I had to set up a few things:
- Registered a blob datastore (`blobkey`) pointing to the `curated` container on `amazondatalake60301919`
- Registered the sampled gold dataset as a data asset in Azure ML
- Created a Feature Store entity (`AmazonReview`) with `asin` and `reviewerID` as index columns
- Had a compute cluster (`lab4-60301919`) already provisioned from earlier

### Pipeline Overview

The pipeline lives in `pipelines/feature_pipeline.yml` and runs these steps in order:

1. **Split** - splits the data into train/val/test (70/15/15) so we dont leak information between sets
2. **Normalize** - runs on each split separately. lowercases text, strips out URLs, numbers, punctuation, and filters reviews that are too short (under 10 chars)
3. **Feature extraction** - four components run on the normalized train set
4. **Merge** - joins all the feature outputs together on the entity keys

### Components

Each component has its own folder under `components/` with a Python script and a `component.yml`.

#### split_dataset
Pretty straightforward - uses sklearn's `train_test_split` twice to get 70% train, 15% val, 15% test. Takes a seed parameter so the split is reproducible.

#### normalize_text
Cleans up the review text before we extract features from it. I used `re` for regex stuff. It does:
- lowercase everything
- remove URLs (http/https/www patterns)
- remove numbers
- strip punctuation
- collapse extra whitespace
- drop reviews under 10 characters since they dont really have enough info to be useful

#### length_features
Simple but useful features:
- **review_length_words** - word count of the review
- **review_length_chars** - character count

I included these because longer reviews tend to be more detailed and could correlate with how helpful a review is or how extreme the rating is. They are cheap to compute and usually end up being decent predictors.

#### sentiment_features
Uses NLTK's VADER sentiment analyzer to get:
- **sentiment_pos** - proportion of positive words
- **sentiment_neg** - proportion of negative words  
- **sentiment_neu** - proportion of neutral words
- **sentiment_compound** - overall sentiment score from -1 (very negative) to +1 (very positive)

I went with VADER because it works well on short social media-style text without needing to train anything. Amazon reviews are kinda similar in tone to tweets and product comments so VADER handles them fine. This component needs a custom conda environment since nltk isnt in the default Azure ML images.

#### tfidf_features
Creates TF-IDF vectors using sklearn's `TfidfVectorizer` with these settings:
- max 500 features (to keep things manageable)
- english stop words removed
- unigrams and bigrams (ngram_range 1,2)

The important thing here is that the vectorizer is **fit only on the training split** and then applied to val and test. If we fit on all the data we'd be leaking info from val/test into the vocabulary which would give us unrealistic performance estimates.

Bigrams help capture phrases like "not good" or "battery life" that you lose with just single words.

#### sbert_embeddings
Uses Sentence-BERT (model: `all-MiniLM-L6-v2`) to generate dense vector embeddings for each review. These capture semantic meaning that TF-IDF cant - like understanding that "great product" and "excellent item" mean basically the same thing even though they share no words.

I picked MiniLM because its small and fast but still gives solid embeddings. The full BERT models would take way too long on a CPU cluster. This component also needs a custom environment with `sentence-transformers` and `torch`.

#### merge_features
Takes the outputs from length, sentiment, tfidf, and sbert components and joins them all on `asin` + `reviewerID`. The result is one parquet file with every feature column.

### Why these features?

I tried to cover different angles of the review text:

| Feature Type | What it captures | Why its useful |
|---|---|---|
| Length | How much someone wrote | Longer reviews might be more informative, short ones might be spam |
| Sentiment | Emotional tone | Directly related to rating prediction, helps with opinion mining |
| TF-IDF | Important words/phrases | Classical text representation, good baseline for most NLP tasks |
| SBERT | Semantic meaning | Captures context and synonyms that bag-of-words methods miss |

The idea was to have a mix of simple hand-crafted features (length, sentiment) and more complex learned representations (TF-IDF, embeddings). This way downstream models can pick whichever signals are most useful.

### Feature Store Registration

After the pipeline ran successfully I registered everything in the Azure ML Feature Store:
- **Entity**: `AmazonReview` (version 1) with `asin` and `reviewerID` as keys
- **Feature Set**: `amazon_review_text_features` (version 1) containing all the engineered features

The feature set spec points to the merge output from the pipeline run. Having this in the Feature Store means I can reuse these exact features in future labs without recomputing them.

### Project Structure

```
├── components/
│   ├── split_dataset/        # Train/val/test splitting
│   ├── normalize_text/       # Text cleaning and filtering
│   ├── length_features/      # Word and char count features
│   ├── sentiment_features/   # VADER sentiment scores (custom env)
│   ├── tfidf_features/       # TF-IDF vectorization
│   ├── sbert_embeddings/     # Sentence-BERT embeddings (custom env)
│   └── merge_features/       # Joins all feature outputs
├── pipelines/
│   └── feature_pipeline.yml  # Azure ML pipeline definition
├── feature_store/
│   ├── entity_amazon_review.yml
│   ├── feature_set_amazon_review_text_features.yml
│   ├── FeatureSetSpec.yaml
│   └── spec/
├── data/
│   └── features_v1_sampled.yml
├── datastores/               # (gitignored - contains account keys)
├── 01_load_and_clean_reviews.ipynb
├── 02_enrich_with_metadata.ipynb
└── 03_write_gold_features_v1.ipynb
```

### Technologies
- **Azure ML** - pipeline orchestration, component registry, compute clusters
- **Azure ML Feature Store** - feature versioning and reuse
- **ADLS / Blob Storage** - data lake for storing datasets
- **scikit-learn** - TF-IDF, train/test splitting
- **NLTK (VADER)** - sentiment analysis
- **Sentence-Transformers** - semantic embeddings
- **pandas / pyarrow** - data manipulation and parquet I/O