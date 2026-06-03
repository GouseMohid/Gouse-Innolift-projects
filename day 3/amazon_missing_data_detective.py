import pandas as pd


# Load the Amazon review dataset.
df = pd.read_csv("data/amazon_reviews_sample.csv")

# Check whether any column contains missing values.
print("Amazon Missing Data Detective")
print("-----------------------------")

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing reviews with text and missing ratings with the average rating.
filled_df = df.fillna(
    {
        "review": "No review provided",
        "rating": df["rating"].mean(),
    }
)

# Create another cleaned version by removing rows that contain missing values.
dropped_df = df.dropna()

# Compare the original, filled, and dropped DataFrame shapes.
print("\nShape before cleaning:")
print(df.shape)

print("\nShape after fillna:")
print(filled_df.shape)

print("\nMissing values after fillna:")
print(filled_df.isnull().sum())

print("\nShape after dropna:")
print(dropped_df.shape)
