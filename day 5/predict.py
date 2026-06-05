from sentiment_model import predict_sentiment


CASES = [
    "The product arrived early, feels sturdy, and works perfectly every day.",
    "It is okay for the price, but the setup was confusing and the box looked old.",
    "Terrible quality. It stopped working after two days and customer support ignored me.",
]


def main():
    for index, review in enumerate(CASES, start=1):
        result = predict_sentiment(review)
        triggers = ", ".join(item["word"] for item in result["trigger_words"]) or "None"
        print(f"Case {index}")
        print(f"Review: {review}")
        print(f"Prediction: {result['sentiment']} ({result['confidence']}%)")
        print(f"Trigger words: {triggers}")
        print()


if __name__ == "__main__":
    main()
