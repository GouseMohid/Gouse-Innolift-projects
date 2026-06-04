import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("data/amazon_reviews_sample.csv")

df["sentiment"] = "Neutral"
df.loc[df["rating"] >= 4, "sentiment"] = "Positive"
df.loc[df["rating"] <= 2, "sentiment"] = "Negative"

sentiment_order = ["Negative", "Neutral", "Positive"]
avg_rating = df.groupby("sentiment")["rating"].mean().reindex(sentiment_order)

mean_rating = df["rating"].mean()
color_list = ["#d64545", "#f2c94c", "#27ae60"]

plt.figure(figsize=(8, 5))

plt.bar(
    avg_rating.index,
    avg_rating.values,
    color=color_list,
    label="Average rating by sentiment",
)

plt.axhline(
    mean_rating,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Mean rating: {mean_rating:.2f}",
)

plt.title("Average Amazon Review Rating by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Rating")
plt.legend()

plt.tight_layout()
plt.show()
