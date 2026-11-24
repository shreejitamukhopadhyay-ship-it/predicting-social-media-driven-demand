# Predicting Social Media-Driven Demand in the Retail Supply Chain

**Academic project · M.Sc. Data Science · Shreejita Mukhopadhyay**

This case study explores whether customer review sentiment could add useful context to fashion retail demand forecasts. My original thesis describes this as “social media-driven demand,” but its sentiment input is customer product reviews, not social media posts. The distinction matters: this work tests an exploratory proxy, not a direct social-media signal.

## Project status

This repository is a public portfolio case study. The thesis source folder contains multiple model versions with conflicting metrics, so the figures below are transcribed from the final presentation and are not yet independently reproduced from one verified pipeline. No raw data, full report, presentations, or private academic identifiers are included.

## Reported results

| Model (final presentation) | MAE | RMSE | MAPE | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression baseline | 2.827 | 3.529 | 78.13% | -1.855 |
| Sentiment-enhanced Linear Regression | 2.877 | 3.505 | 80.32% | -1.816 |
| Random Forest | 1.633 | 2.106 | 44.03% | -0.017 |
| XGBoost | 1.624 | 2.094 | 44.43% | -0.005 |

In that presentation, tree models had lower average errors than the linear models, but their R² values were around zero and slightly negative. Adding sentiment to the linear model slightly improved RMSE, while MAE and MAPE became worse. These results do not establish reliable forecasting performance or show that sentiment itself caused an improvement.

## Data and approach

The thesis combines fashion retail transactions with Women's E-Commerce Clothing Reviews. The sales records are summarized monthly by item; review text is scored with VADER and summarized by category; a manual mapping joins sales item names to review categories. The presentation reports that 32 of 50 sales items were mapped.

- [Fashion Retail Sales — A. Soundankar (Kaggle)](https://www.kaggle.com/datasets/atharvasoundankar/fashion-retail-sales)
- [Women's E-Commerce Clothing Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Review records have no timestamps, so the review sentiment cannot be aligned to the time a real forecast would have been made. Customer reviews are only a proxy for social media activity. The current Fashion Retail Sales license and terms for redistribution or derived materials must be confirmed before publishing data or row-level outputs. The raw datasets are intentionally not included.

## Code

The scripts folder contains source-derived scripts for cleaning the sales data, cleaning review text, and scoring reviews with VADER. They are included as academic project code, not as a verified reproduction of the final presentation's model results. The result versions must be reconciled before a single final modelling pipeline can be published.

Download each dataset from its cited page and place it locally in `data/raw/` using the filenames expected by the scripts:

- `Fashion_Retail_Sales.csv`
- `Womens Clothing E-Commerce Reviews data.csv`

Install dependencies with `python -m pip install -r requirements.txt`. From the repository root, run `python scripts/clean_fashion_retail_sales.py`, `python scripts/clean_womens_clothing_reviews.py`, and then `python scripts/sentiment_analysis_vader.py`. The review scripts may download NLTK resources on first use. Generated and raw data paths are excluded by `.gitignore`.

## Reconciliation needed

A later lagged-model workbook reports different metrics from the final presentation, and an old dummy report contains another comparison. The later script predicts next-month transactions but appears to split by the current row's date; its target-month coverage needs verification. These outputs are intentionally not combined. A reproducible, single final pipeline and matching result table should be added after that reconciliation.

## Next version

1. Select and document the academically intended final pipeline.
2. Ensure every feature is available before the forecast month; review sentiment must have timestamps to serve as a genuine early signal.
3. Compare with simple last-period and seasonal-naive baselines over several rolling test windows.
4. Report the test dates, row counts, metric definitions, mapping coverage, and limitations.

This repository describes academic work and does not contain employer or customer data from employment.
