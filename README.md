# Boston Housing Price Predictor

Predicting median home value (`MEDV`) from property and neighborhood features using the classic Boston Housing dataset.

## What's in here

- Data cleaning: dropping 16 rows where price was artificially capped at $50,000 (a known data collection artifact, not real sale prices)
- EDA: distribution plots, correlation heatmap
- Feature engineering: log transforms on skewed columns (`CRIM`, `DIS`, `LSTAT`), plus a couple of interaction features (`RM_LSTAT`, `TAX_per_room`, `DIS_RAD`)
- Baseline comparison (predicting the average price for everything) to sanity-check the model is actually learning something
- Linear Regression as the main model, with a Gradient Boosting comparison at the end
- Evaluation with R², RMSE, a predicted-vs-actual plot, and a residual plot

## Results

| Model | R² | RMSE |
|---|---|---|
| Baseline (mean) | -0.08 | 7.41 |
| Linear Regression | 0.82 | 3.02 |
| Gradient Boosting | 0.89 | 2.41 |

RMSE is in $1000s. Linear Regression is off by about $3,000 on average, roughly 15% of the average home price in the dataset.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook Boston_Housing_Price_Predictor.ipynb
```

Keep `boston.csv` in the same folder as the notebook — it's loaded with a relative path.

## A note on the data

This is the classic Boston Housing dataset from a 1978 study. It includes a column (`B`) that measures the proportion of Black residents by town, which was originally included as a proxy for a socioeconomic effect. It's been widely criticized for baking racial bias into the data, and scikit-learn removed the dataset from its library in 2020 for this reason. It's used here as a well-known teaching dataset for practicing regression, not as an endorsement of the original methodology.
