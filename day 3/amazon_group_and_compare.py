import pandas as pd


# Load the Amazon review dataset.
df = pd.read_csv("data/amazon_reviews_sample.csv")

# Create sentiment groups from the rating column.
df["sentiment"] = "Neutral"
df.loc[df["rating"] >= 4, "sentiment"] = "Positive"
df.loc[df["rating"] <= 2, "sentiment"] = "Negative"

# Compare review groups using groupby and nlargest.
print("Amazon Group and Compare")
print("------------------------")

print("Average rating by sentiment:")
print(df.groupby("sentiment")["rating"].mean())

print("\nReview count by sentiment:")
print(df.groupby("sentiment")["rating"].count())

print("\nTop 3 highest rated reviews:")
print(df.nlargest(3, "rating")[["review", "rating", "sentiment"]])
