import pandas as pd
import numpy as np


# Load the Amazon review dataset from the project data folder.
reviews = pd.read_csv("data/amazon_reviews_sample.csv")

# Convert the rating column into a NumPy array for numerical analysis.
ratings = np.array(reviews["rating"])

# Use boolean filters to separate ratings into sentiment-style groups.
positive_ratings = ratings[ratings >= 4]
neutral_ratings = ratings[ratings == 3]
negative_ratings = ratings[ratings <= 2]

# Print a simple summary report for the ratings.
print("Amazon Review NumPy Rating Analyser")
print("-----------------------------------")
print(f"Ratings array: {ratings}")
print(f"Total reviews: {len(ratings)}")
print(f"Mean rating: {ratings.mean():.2f}")
print(f"Highest rating: {ratings.max()}")
print(f"Lowest rating: {ratings.min()}")
print(f"Positive reviews: {len(positive_ratings)}")
print(f"Neutral reviews: {len(neutral_ratings)}")
print(f"Negative reviews: {len(negative_ratings)}")
