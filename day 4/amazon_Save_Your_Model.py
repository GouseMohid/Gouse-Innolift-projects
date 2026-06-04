import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression


def create_features(review):
    words = review.split()

    return {
        "review_length": len(words),
        "char_count": len(review),
        "word_avg_length": sum(len(word) for word in words) / len(words),
        "exclamation_count": review.count("!"),
    }


df = pd.read_csv("data/amazon_reviews_sample.csv")

feature_rows = [create_features(review) for review in df["review"]]
features_df = pd.DataFrame(feature_rows)

X = features_df
y = df["rating"]

model = LinearRegression()
model.fit(X, y)

model_path = "day 4/amazon_rating_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved successfully as {model_path}")

with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)

new_reviews = [
    "This product is excellent and works perfectly. I love it!",
    "The item is okay but the quality could be better.",
    "Terrible product. It stopped working after one day!",
]

new_features = pd.DataFrame([create_features(review) for review in new_reviews])
predictions = loaded_model.predict(new_features)

print("\nPredicted Ratings")
print("-----------------")

for review, rating in zip(new_reviews, predictions):
    rating = max(1, min(5, rating))

    print(f"Review: {review}")
    print(f"Predicted rating: {rating:.2f}")
    print()
