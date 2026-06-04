import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/amazon_reviews_sample.csv")

df["review_length"] = df["review"].str.split().str.len()
df["char_count"] = df["review"].str.len()

numeric_df = df.select_dtypes(include="number")

X = numeric_df.drop("rating", axis=1)
y = numeric_df["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Your First Regression")
print("---------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

new_review = pd.DataFrame(
    {
        "review_length": [18],
        "char_count": [95],
    }
)

predicted_rating = model.predict(new_review)

print("\nPredicted rating for new product review:")
print(f"{predicted_rating[0]:.2f}")
