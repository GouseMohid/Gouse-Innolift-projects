import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/amazon_reviews_sample.csv")

df["review_length"] = df["review"].str.split().str.len()
df["char_count"] = df["review"].str.len()
df["word_avg_length"] = df["review"].apply(
    lambda text: sum(len(word) for word in text.split()) / len(text.split())
)
df["exclamation_count"] = df["review"].str.count("!")

features = ["review_length", "char_count", "word_avg_length", "exclamation_count"]
target = "rating"

results = []

print("Feature Comparison")
print("------------------")

for feature in features:
    X = df[[feature]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append(
        {
            "feature": feature,
            "rmse": rmse,
        }
    )

    print(f"{feature} RMSE: {rmse:.2f}")

best_feature = min(results, key=lambda item: item["rmse"])

print("\nBest single feature:")
print(f"{best_feature['feature']} with RMSE {best_feature['rmse']:.2f}")
