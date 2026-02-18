# DSAI3202 Lab 3 - Data Preprocessing in Azure

This lab is a step in Assignment 1. It focuses on building a clean, enriched, and feature-ready dataset using Spark and Databricks.

## What I did in this lab
- Loaded raw review data into Spark, inspected schema, and handled missing/invalid values.
- Cleaned and standardized text and rating fields so they are consistent for downstream processing.
- Enriched reviews with metadata to add context needed for feature creation.
- Wrote curated outputs using Parquet and Delta formats for reliable reuse in later assignments.

## Technologies used
- **Apache Spark**: Distributed data processing engine used for reading, cleaning, transforming, and joining large datasets.
- **Databricks**: Managed Spark environment where the notebooks run. It provides clusters, notebooks, and optimized storage access.
- **Parquet**: Columnar storage format used for efficient reads and writes, especially for analytics workloads.
- **Delta Lake (Delta)**: Storage layer on top of Parquet that adds ACID transactions, schema enforcement, and time travel.

## Notebook walkthrough

### 01_load_and_clean_reviews.ipynb
Purpose: Load raw reviews and produce a clean base dataset.

Key steps:
- Read the raw reviews data into a Spark DataFrame.
- Inspect data types and basic statistics.
- Remove or fix nulls, duplicates, and invalid values.
- Standardize columns (for example, trimming strings and normalizing ratings).
- Write the cleaned output to storage for later steps.

### 02_enrich_with_metadata.ipynb
Purpose: Combine the cleaned reviews with additional metadata.

Key steps:
- Read the cleaned reviews and metadata datasets.
- Join on shared keys (such as product or business identifiers).
- Select and rename columns to keep only the needed fields.
- Validate joins and record counts after enrichment.
- Write the enriched dataset in a durable format (Parquet or Delta).

### 03_write_gold_features_v1.ipynb
Purpose: Build a feature-ready dataset for analytics or ML.

Key steps:
- Read the enriched dataset.
- Create feature columns (for example, aggregations, normalized scores, or derived text metrics).
- Apply final filtering and schema cleanup.
- Save the gold dataset using Delta for versioned, reliable storage.

## Output artifacts
- **Cleaned reviews**: Base dataset after initial cleaning.
- **Enriched reviews**: Cleaned reviews joined with metadata.
- **Gold features (v1)**: Final feature set stored as Delta for Assignment 1.