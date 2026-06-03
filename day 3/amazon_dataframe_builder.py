import pandas as pd


# Load the Amazon review dataset from the CSV file.
dataset = pd.read_csv("data/amazon_reviews_sample.csv")

# Build a DataFrame manually from the dataset columns.
df = pd.DataFrame(
    {
        "review": dataset["review"],
        "rating": dataset["rating"],
    }
)

# Add a new sentiment column based on the rating value.
df["sentiment"] = "Neutral"
df.loc[df["rating"] >= 4, "sentiment"] = "Positive"
df.loc[df["rating"] <= 2, "sentiment"] = "Negative"

# Print the DataFrame preview and basic details.
print("Amazon Review DataFrame Builder")
print("--------------------------------")

print("Head:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nSentiment counts:")
print(df["sentiment"].value_counts())
