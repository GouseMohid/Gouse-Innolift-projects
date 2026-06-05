# Product Review Sentiment Analyzer

## Problem Statement

This project predicts Amazon product review sentiment as Negative, Neutral, or Positive from review text. Ratings are used to create the target label:

- 1-2: Negative
- 3: Neutral
- 4-5: Positive

## Dataset Source

Use the assigned Amazon reviews CSV from Kaggle/UCI and place it at:

```text
day 5/data/amazon_reviews.csv
```

A small fallback demo file is included at:

```text
day 5/data/amazon_reviews_sample.csv
```

## Features Used

- `review`: raw product review text
- `cleaned_review`: lowercase text with HTML, URLs, punctuation, numbers, extra spaces, and stopwords removed
- `rating`: numeric score used to create the target label
- `text_length`: review length used for EDA charts

## Algorithm

`train.py` uses:

- `TfidfVectorizer` for text features
- `RandomForestClassifier` for sentiment classification
- 80/20-style train-test splitting with stratification when the dataset is large enough

## Accuracy

On the included 10-row sample dataset, the model reached 50% accuracy. This sample is only for demo execution; use the full assigned dataset to get a more meaningful score and try to beat 70%.

## Project Files

```text
day 5/
|-- data/
|   |-- amazon_reviews.csv              full dataset goes here
|   |-- amazon_reviews_sample.csv       fallback demo dataset
|   |-- amazon_reviews_m1_cleaned.csv   generated cleaned output
|-- charts/
|   |-- sentiment_distribution.png
|   |-- rating_by_sentiment.png
|   |-- correlation_heatmap.png
|-- models/
|   |-- model.pkl
|   |-- tfidf_vectorizer.pkl
|-- model.py
|-- train.py
|-- predict.py
|-- requirements.txt
|-- README.md
```

## How To Run

```powershell
cd "day 5"
pip install -r requirements.txt
python model.py
python train.py
python predict.py
```

## What Each Script Does

- `model.py`: loads the CSV, prints shape/head/dtypes, runs EDA, cleans missing values, verifies 0 nulls, and saves cleaned data.
- `train.py`: repeats EDA checks, trains the RandomForest model, prints accuracy/report, saves `model.pkl`, and creates three chart PNGs.
- `predict.py`: loads `model.pkl` and `tfidf_vectorizer.pkl`, then predicts three real-world review examples.
