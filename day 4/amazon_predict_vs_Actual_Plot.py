import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/amazon_reviews_sample.csv")

df["review_length"] = df["review"].str.split().str.len()
df["char_count"] = df["review"].str.len()
df["word_avg_length"] = df["review"].apply(
    lambda text: sum(len(word) for word in text.split()) / len(text.split())
)
df["exclamation_count"] = df["review"].str.count("!")

X = df[["review_length", "char_count", "word_avg_length", "exclamation_count"]]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    label="Predicted ratings",
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect prediction",
)

plt.title("Actual vs Predicted Amazon Review Ratings")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.legend()

plt.tight_layout()
plt.show()
