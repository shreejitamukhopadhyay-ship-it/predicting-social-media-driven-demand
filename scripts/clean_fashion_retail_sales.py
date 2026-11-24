"""
Preprocess the Fashion Retail Sales dataset for the MSc Data Science project:
"Predicting Social Media-Driven Flash Demand in Retail Supply Chain".

Required packages:
    pip install pandas numpy openpyxl

Run this script from the project root:
    python clean_fashion_retail_sales.py
"""

from pathlib import Path

import numpy as np
import openpyxl  # Imported because pandas uses openpyxl when saving .xlsx files.
import pandas as pd


# Repository root is the parent of this scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the input CSV file path.
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Fashion_Retail_Sales.csv"

# Define the output folder and output file paths.
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"
CLEANED_CSV = OUTPUT_FOLDER / "cleaned_fashion_retail_sales.csv"
CLEANED_EXCEL = OUTPUT_FOLDER / "cleaned_fashion_retail_sales.xlsx"
ITEM_SUMMARY_CSV = OUTPUT_FOLDER / "item_sales_summary.csv"
ITEM_SUMMARY_EXCEL = OUTPUT_FOLDER / "item_sales_summary.xlsx"
MONTHLY_SALES_CSV = OUTPUT_FOLDER / "monthly_item_sales.csv"
MONTHLY_SALES_EXCEL = OUTPUT_FOLDER / "monthly_item_sales.xlsx"

# These are the columns expected after basic column-name cleaning.
EXPECTED_COLUMNS = [
    "customer_reference_id",
    "item_purchased",
    "purchase_amount_usd",
    "date_purchase",
    "review_rating",
    "payment_method",
]


def print_step(message):
    """Print a clear message so beginners can follow the script progress."""

    print(f"\n--- {message} ---")


def clean_column_names(df):
    """Clean column names by stripping spaces, replacing spaces, and using lowercase."""

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.lower()
    )
    return df


def check_expected_columns(df):
    """Check that all required columns are available before cleaning starts."""

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]

    if missing_columns:
        available_columns = ", ".join(df.columns)
        missing_columns_text = ", ".join(missing_columns)
        raise KeyError(
            f"Missing expected column(s): {missing_columns_text}. "
            f"Available columns are: {available_columns}"
        )


def parse_dates_safely(date_series):
    """Parse dates that may be day-month-year or month-day-year."""

    # First try day-month-year parsing because the dataset contains values like 23-03-2023.
    dates_day_first = pd.to_datetime(date_series, errors="coerce", dayfirst=True)

    # Then try month-day-year parsing for any dates that could not be parsed the first time.
    dates_month_first = pd.to_datetime(date_series, errors="coerce", dayfirst=False)

    # Keep the day-first result where it worked, and use month-first as a fallback.
    return dates_day_first.fillna(dates_month_first)


def get_most_used_payment_method(payment_methods):
    """Return the most common payment method for one item category."""

    modes = payment_methods.mode()

    if modes.empty:
        return "Unknown"

    return modes.iloc[0]


def main():
    """Load, inspect, clean, summarize, and save the Fashion Retail Sales dataset."""

    try:
        print_step("Step 1: Imported required libraries")
        print("Libraries imported: pandas, numpy, openpyxl")
        print("NumPy version:", np.__version__)
        print("openpyxl version:", openpyxl.__version__)

        print_step("Step 2: Loading the dataset")

        # Check that the input CSV file exists before trying to load it.
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        # Load the CSV dataset into a pandas DataFrame.
        df = pd.read_csv(INPUT_FILE)
        print(f"Dataset loaded successfully from: {INPUT_FILE}")

        print_step("Step 3: Displaying the original dataset information")
        print("\nFirst 5 rows:")
        print(df.head())

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nDataset shape:")
        print(df.shape)

        print("\nMissing value count:")
        print(df.isna().sum())

        print("\nDuplicate row count:")
        print(df.duplicated().sum())

        print("\nBasic data information:")
        df.info()

        print_step("Step 4: Cleaning column names")
        df = clean_column_names(df)
        print("Cleaned column names:")
        print(df.columns.tolist())

        print_step("Step 5: Renaming columns clearly")

        # Rename columns into the final required names.
        # After the cleaning step, "Purchase Amount (USD)" becomes "purchase_amount_usd".
        df = df.rename(
            columns={
                "customer_reference_id": "customer_reference_id",
                "item_purchased": "item_purchased",
                "purchase_amount_usd": "purchase_amount_usd",
                "date_purchase": "date_purchase",
                "review_rating": "review_rating",
                "payment_method": "payment_method",
            }
        )

        check_expected_columns(df)
        df = df[EXPECTED_COLUMNS]
        print("All expected columns were found.")

        print_step("Step 6: Removing duplicate rows")
        duplicate_count_before = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"Removed {duplicate_count_before} duplicate row(s).")

        print_step("Step 7: Handling missing values")

        # Convert numeric columns before calculating medians.
        df["purchase_amount_usd"] = pd.to_numeric(
            df["purchase_amount_usd"], errors="coerce"
        )
        df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")

        rows_before_item_drop = len(df)
        df = df.dropna(subset=["item_purchased"])
        print(f"Removed {rows_before_item_drop - len(df)} row(s) with missing items.")

        rows_before_date_drop = len(df)
        df = df.dropna(subset=["date_purchase"])
        print(f"Removed {rows_before_date_drop - len(df)} row(s) with missing dates.")

        # Fill missing purchase amounts using the median amount for the same item.
        df["purchase_amount_usd"] = df["purchase_amount_usd"].fillna(
            df.groupby("item_purchased")["purchase_amount_usd"].transform("median")
        )

        # If a whole item category has no purchase amount median, use the overall median.
        df["purchase_amount_usd"] = df["purchase_amount_usd"].fillna(
            df["purchase_amount_usd"].median()
        )

        # Fill missing review ratings using the median rating for the same item.
        df["review_rating"] = df["review_rating"].fillna(
            df.groupby("item_purchased")["review_rating"].transform("median")
        )

        # If a whole item category has no review rating median, use the overall median.
        df["review_rating"] = df["review_rating"].fillna(df["review_rating"].median())

        # Fill missing payment methods with a clear label.
        df["payment_method"] = df["payment_method"].fillna("Unknown")

        print("Missing values after cleaning:")
        print(df.isna().sum())

        print_step("Step 8: Converting date_purchase into datetime format")
        df["date_purchase"] = parse_dates_safely(df["date_purchase"])

        invalid_date_count = df["date_purchase"].isna().sum()
        if invalid_date_count > 0:
            df = df.dropna(subset=["date_purchase"])
            print(f"Removed {invalid_date_count} row(s) with invalid date values.")
        else:
            print("All date values were parsed successfully.")

        print_step("Step 9: Creating time-based columns")
        df["year"] = df["date_purchase"].dt.year
        df["month"] = df["date_purchase"].dt.month
        df["month_name"] = df["date_purchase"].dt.month_name()
        df["quarter"] = df["date_purchase"].dt.quarter
        df["day_of_week"] = df["date_purchase"].dt.day_name()
        df["week_of_year"] = df["date_purchase"].dt.isocalendar().week.astype(int)
        print("Created year, month, month_name, quarter, day_of_week, and week_of_year.")

        print_step("Step 10: Cleaning text columns")
        df["item_purchased"] = (
            df["item_purchased"].astype(str).str.strip().str.lower().str.title()
        )
        df["payment_method"] = df["payment_method"].astype(str).str.strip().str.title()
        print("Cleaned item_purchased and payment_method text values.")

        print_step("Step 11: Creating item-level summary dataset")
        item_sales_summary = (
            df.groupby("item_purchased")
            .agg(
                total_sales_amount=("purchase_amount_usd", "sum"),
                average_sales_amount=("purchase_amount_usd", "mean"),
                total_transactions=("purchase_amount_usd", "size"),
                average_review_rating=("review_rating", "mean"),
                most_used_payment_method=(
                    "payment_method",
                    get_most_used_payment_method,
                ),
            )
            .reset_index()
        )
        print("Item-level summary dataset created.")

        print_step("Step 12: Creating monthly item-level sales dataset")
        monthly_item_sales = (
            df.groupby(["year", "month", "item_purchased"])
            .agg(
                monthly_sales_amount=("purchase_amount_usd", "sum"),
                monthly_transactions=("purchase_amount_usd", "size"),
                average_monthly_rating=("review_rating", "mean"),
            )
            .reset_index()
            .sort_values(["year", "month", "item_purchased"])
        )
        print("Monthly item-level sales dataset created.")

        print_step("Step 13: Saving all output files")

        # Create the output folder if it does not already exist.
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save the cleaned full dataset.
        df.to_csv(CLEANED_CSV, index=False)
        df.to_excel(CLEANED_EXCEL, index=False)

        # Save the item-level summary dataset.
        item_sales_summary.to_csv(ITEM_SUMMARY_CSV, index=False)
        item_sales_summary.to_excel(ITEM_SUMMARY_EXCEL, index=False)

        # Save the monthly item-level sales dataset.
        monthly_item_sales.to_csv(MONTHLY_SALES_CSV, index=False)
        monthly_item_sales.to_excel(MONTHLY_SALES_EXCEL, index=False)

        print(f"Cleaned CSV saved to: {CLEANED_CSV}")
        print(f"Cleaned Excel file saved to: {CLEANED_EXCEL}")
        print(f"Item summary CSV saved to: {ITEM_SUMMARY_CSV}")
        print(f"Item summary Excel file saved to: {ITEM_SUMMARY_EXCEL}")
        print(f"Monthly item sales CSV saved to: {MONTHLY_SALES_CSV}")
        print(f"Monthly item sales Excel file saved to: {MONTHLY_SALES_EXCEL}")

        print_step("Preprocessing completed successfully")
        print("The dataset is ready for further flash-demand analysis.")

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

