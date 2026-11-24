"""
Clean the Kaggle Womens Clothing E-Commerce Reviews dataset.

Required packages:
    pip install pandas nltk openpyxl

Run this script from the project root:
    python clean_womens_clothing_reviews.py
"""

from pathlib import Path
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Repository root is the parent of this scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the input CSV file path.
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Womens Clothing E-Commerce Reviews data.csv"

# Define the output folder and output file paths.
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = OUTPUT_FOLDER / "cleaned_reviews.csv"
OUTPUT_EXCEL = OUTPUT_FOLDER / "cleaned_reviews.xlsx"

# Define the dataset column that contains the customer review text.
REVIEW_COLUMN = "Review Text"


def download_nltk_resources():
    """Download the NLTK resources needed for stopword removal and tokenization."""

    # Download the English stopwords list, such as "the", "and", "is", and "in".
    nltk.download("stopwords")

    # Download the tokenizer model used by word_tokenize.
    nltk.download("punkt")

    # Download punkt_tab for compatibility with newer NLTK versions.
    nltk.download("punkt_tab")


def clean_review_text(review_text, english_stopwords, punctuation_table):
    """Clean one review by lowercasing, removing punctuation, tokenizing, and removing stopwords."""

    # Convert the review to lowercase so words like "Great" and "great" are treated the same.
    review_text = review_text.lower()

    # Remove punctuation marks such as commas, full stops, brackets, and exclamation marks.
    review_text = review_text.translate(punctuation_table)

    # Split the cleaned sentence into individual words or tokens.
    tokens = word_tokenize(review_text)

    # Keep only alphabetic words and remove common English stopwords.
    cleaned_tokens = [
        token
        for token in tokens
        if token.isalpha() and token not in english_stopwords
    ]

    # Join the remaining useful words back into one cleaned review string.
    return " ".join(cleaned_tokens)


def main():
    """Load, clean, preview, and save the Womens Clothing E-Commerce Reviews dataset."""

    try:
        # Download the NLTK data required by this script before text cleaning starts.
        download_nltk_resources()

        # Check that the input CSV file exists before trying to load it.
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        # Load the CSV dataset into a pandas DataFrame.
        df = pd.read_csv(INPUT_FILE)

        # Display the original number of rows and columns before cleaning.
        print("Dataset shape before cleaning:", df.shape)

        # Check that the review text column exists before trying to clean it.
        if REVIEW_COLUMN not in df.columns:
            available_columns = ", ".join(df.columns)
            raise KeyError(
                f'Missing required column "{REVIEW_COLUMN}". '
                f"Available columns are: {available_columns}"
            )

        # Remove rows where the Review Text column has missing or null values.
        df = df.dropna(subset=[REVIEW_COLUMN])

        # Remove duplicate rows from the full dataset.
        df = df.drop_duplicates()

        # Create a set of English stopwords for faster lookup during cleaning.
        english_stopwords = set(stopwords.words("english"))

        # Create a translation table that removes punctuation from text.
        punctuation_table = str.maketrans("", "", string.punctuation)

        # Apply the text cleaning function to each review and save the result in a new column.
        df["Cleaned_Review"] = df[REVIEW_COLUMN].apply(
            lambda text: clean_review_text(text, english_stopwords, punctuation_table)
        )

        # Display the number of rows and columns after cleaning.
        print("Dataset shape after cleaning:", df.shape)

        # Display the first five rows so beginners can preview the cleaned output.
        print("\nSample output preview:")
        print(df.head())

        # Create the output folder if it does not already exist.
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save the cleaned dataset as a CSV file.
        df.to_csv(OUTPUT_CSV, index=False)

        # Save the cleaned dataset as an Excel file.
        df.to_excel(OUTPUT_EXCEL, index=False)

        # Confirm where the cleaned files were saved.
        print(f"\nCleaned CSV saved to: {OUTPUT_CSV}")
        print(f"Cleaned Excel file saved to: {OUTPUT_EXCEL}")

    except FileNotFoundError as error:
        # Show a clear error message if the input file path is wrong or missing.
        print(f"File error: {error}")

    except KeyError as error:
        # Show a clear error message if an expected column is missing.
        print(f"Column error: {error}")

    except Exception as error:
        # Show a general error message for any other unexpected issue.
        print(f"An unexpected error occurred: {error}")


# Run the main function only when this file is executed directly.
if __name__ == "__main__":
    main()

