import pandas as pd


# Load the Amazon review CSV file into a DataFrame.
df = pd.read_csv("data/amazon_reviews_sample.csv")

# Print basic CSV details and sample rows.
print("Amazon Review CSV Explorer")
print("--------------------------")

print("Shape:")
print(df.shape)

print("\nColumn names:")
print(list(df.columns))

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 3 rows:")
print(df.tail(3))

# Count how many reviews are available for each rating.
print("\nRating counts:")
print(df["rating"].value_counts().sort_index())
