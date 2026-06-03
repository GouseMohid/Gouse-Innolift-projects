import pandas as pd


def eda_report(df):
    # Print the overall size, columns, and data types.
    print("Amazon Full EDA Report")
    print("----------------------")

    print("Shape:")
    print(df.shape)

    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    # Separate numeric and text columns for different summaries.
    numeric_columns = df.select_dtypes(include="number")
    text_columns = df.select_dtypes(include="object")

    print("\nNumeric summary:")
    print(numeric_columns.describe())

    print("\nText columns:")
    print(list(text_columns.columns))

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    # Add sentiment labels from rating values.
    df["sentiment"] = "Neutral"
    df.loc[df["rating"] >= 4, "sentiment"] = "Positive"
    df.loc[df["rating"] <= 2, "sentiment"] = "Negative"

    # Print grouped summaries and top reviews.
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nAverage rating by sentiment:")
    print(df.groupby("sentiment")["rating"].mean())

    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nTop 3 highest rated reviews:")
    print(df.nlargest(3, "rating")[["review", "rating", "sentiment"]])


# Load the dataset and run the EDA report function.
reviews = pd.read_csv("data/amazon_reviews_sample.csv")
eda_report(reviews)
