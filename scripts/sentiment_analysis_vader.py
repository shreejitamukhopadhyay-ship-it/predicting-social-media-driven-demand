"""
Sentiment Analysis using VADER

Installation instructions:
    pip install pandas nltk openpyxl

Run this script from the project root:
    python sentiment_analysis_vader.py
"""

from pathlib import Path

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# Download the VADER lexicon. This is needed for sentiment analysis.
nltk.download("vader_lexicon")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Load the cleaned reviews Excel file.
df = pd.read_excel(PROCESSED_DIR / "cleaned_reviews.xlsx")


# This is the cleaned review text column used for sentiment analysis.
review_column = "Cleaned_Review"


# Remove rows where the review text is empty.
df = df.dropna(subset=[review_column])
df = df[df[review_column].astype(str).str.strip() != ""]


# Create the VADER sentiment analyzer.
analyzer = SentimentIntensityAnalyzer()


# Function to get all VADER sentiment scores for one review.
def get_scores(review):
    return analyzer.polarity_scores(str(review))


# Apply VADER sentiment analysis to each review.
scores = df[review_column].apply(get_scores)


# Add the VADER scores as new columns.
df["Positive_Score"] = scores.apply(lambda score: score["pos"])
df["Negative_Score"] = scores.apply(lambda score: score["neg"])
df["Neutral_Score"] = scores.apply(lambda score: score["neu"])
df["Compound_Score"] = scores.apply(lambda score: score["compound"])


# Function to classify each review using the compound score.
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Add the final sentiment label.
df["Sentiment"] = df["Compound_Score"].apply(classify_sentiment)


# Save the final dataset with sentiment scores.
df.to_excel(PROCESSED_DIR / "reviews_with_sentiment.xlsx", index=False)


# Print the first 10 rows showing the review text, compound score, and sentiment.
print(df[["Review Text", "Compound_Score", "Sentiment"]].head(10))

print("Sentiment analysis completed successfully.")

