# Amazon Reviews Sentiment Analysis

## Project Overview

The Amazon Reviews Sentiment Analysis project is a Machine Learning application that predicts customer sentiment from product reviews. The project analyzes customer review text, review summaries, helpfulness metrics, and ratings to classify reviews as Positive or Negative.

The goal of this project is to help businesses understand customer feedback, monitor product performance, and improve customer satisfaction through automated sentiment analysis.

---

## Problem Statement

E-commerce platforms receive thousands of customer reviews daily. Analyzing these reviews manually is time-consuming and inefficient.

This project aims to build a Machine Learning model that automatically predicts customer sentiment based on review data, helping businesses quickly identify positive and negative customer opinions.

---

## Dataset Source

**Dataset Name:** Amazon Fine Food Reviews Dataset

**Source:** Kaggle

The dataset contains customer reviews, ratings, review summaries, helpfulness information, and review text collected from Amazon products.

---

## Features Used (X)

The following features were used as input variables:

* Summary
* Text
* HelpfulnessNumerator
* HelpfulnessDenominator

The textual features were converted into numerical representations using TF-IDF Vectorization.

---

## Target Variable (Y)

### Sentiment

The sentiment target variable was created using the Score column:

* Score > 3 → Positive
* Score ≤ 3 → Negative

The model predicts whether a review expresses positive or negative sentiment.

---

## Data Preprocessing

The following preprocessing steps were performed:

* Loaded dataset using Pandas
* Explored dataset structure and dimensions
* Identified and handled missing values
* Removed duplicate records
* Generated sentiment labels from review scores
* Cleaned and normalized review text
* Converted text into numerical vectors using TF-IDF Vectorizer
* Split data into training and testing sets

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand customer review patterns and sentiment distribution.

### Analysis Performed

* Dataset Shape Analysis
* Missing Value Analysis
* Sentiment Distribution Analysis
* Review Length Analysis
* Statistical Summary
* Feature Correlation Analysis

### Visualizations Created

* Sentiment Distribution Chart
* Review Length Histogram
* Word Frequency Analysis
* Correlation Heatmap
* Confusion Matrix

---

## Algorithm Used: Logistic Regression

Logistic Regression was selected because it is one of the most effective and widely used algorithms for binary classification problems such as sentiment analysis. The algorithm predicts the probability of a review belonging to either the Positive or Negative sentiment class.

It performs exceptionally well with TF-IDF vectorized text data and provides a strong balance between accuracy, efficiency, and interpretability.

---

## Model Evaluation

The model was evaluated using:

### Accuracy Score

Measures the percentage of correctly classified reviews.

### Precision

Measures how many predicted positive reviews were actually positive.

### Recall

Measures how many actual positive reviews were correctly identified.

### F1 Score

Provides a balanced evaluation between precision and recall.

---

## Accuracy Achieved

* Accuracy: XX.XX%
* Precision: XX.XX%
* Recall: XX.XX%
* F1 Score: XX.XX%

*(Replace the above values with your actual model results.)*

---

## Why Logistic Regression?

* Well-suited for binary classification tasks.
* Performs effectively on large text datasets.
* Works efficiently with TF-IDF feature vectors.
* Produces interpretable prediction probabilities.
* Fast training and prediction times.
* Handles high-dimensional textual data effectively.
* Achieved strong sentiment classification performance on Amazon review data.
* Selected as the final model because of its balance between accuracy, speed, and simplicity.

---

## Project Structure

Amazon_Reviews_Sentiment_Analysis/

│

├── amazon_reviews.csv

├── sentiment_analysis.py

├── predict.py

├── best_model.pkl

├── vectorizer.pkl

├── visualize.py

├── requirements.txt

├── README.md

│

├── sentiment_distribution.png

├── confusion_matrix.png

├── review_length_histogram.png

└── word_frequency_chart.png

---

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python sentiment_analysis.py
```

This will:

* Load the dataset
* Preprocess review text
* Train the Logistic Regression model
* Evaluate performance
* Save best_model.pkl

### Step 3: Predict New Reviews

```bash
python predict.py
```

This will:

* Load the saved model
* Predict sentiment for new customer reviews

### Step 4: Generate Visualizations

```bash
python visualize.py
```

This will generate:

* sentiment_distribution.png
* confusion_matrix.png
* review_length_histogram.png
* word_frequency_chart.png

---

## Real-World Applications

* Customer Feedback Analysis
* Product Review Monitoring
* E-Commerce Analytics
* Brand Reputation Management
* Customer Experience Analysis
* Market Research
* Automated Review Classification

---

## Conclusion

This project demonstrates how Machine Learning and Natural Language Processing (NLP) can be used to automatically analyze customer reviews and predict sentiment. The Logistic Regression model effectively classifies reviews as Positive or Negative, helping businesses gain valuable insights from customer feedback and make informed decisions to improve products and services.
