# Boston Housing Price Predictor

Predicting median home value (`MEDV`) from property and neighborhood features, using the classic Boston Housing dataset — from raw data to a live public app.

**Live app:** https://bostonhousingpricepredictor-ppnmoe7bpabxn3h3uu7keu.streamlit.app

## What's in this repo

```
Boston_Housing_Price_Predictor/
├── README.md
├── Boston_Housing_Dashboard.xlsx        # Excel dashboard version of the analysis
├── Boston_Housing_Project.pptx          # Project walkthrough presentation
└── Boston_House_Prediction/
    ├── app.py                           # The Streamlit app
    ├── train_model.py                   # Model training logic (not used by the deployed app)
    ├── boston.csv                       # The dataset
    ├── requirements.txt                 # Python dependencies
    ├── model.pkl                        # Unused — app trains fresh on startup instead
    └── feature_names.pkl                # Unused — app trains fresh on startup instead
```

## The problem with the raw data

The dataset caps 16 house prices at exactly $50,000 — not real sale prices, just how the original 1970s study recorded anything above that threshold. Left in, this artifact drags down model accuracy. It gets dropped during cleaning.

Several features (`CRIM`, `DIS`, `LSTAT`) are heavily skewed, so they're log-transformed to make their relationship with price closer to linear — this helps Linear Regression directly and gives tree models cleaner splits too.

## What was tried, and what actually worked

| Change | Effect |
|---|---|
| Linear Regression, raw data | R² = 0.67 (baseline) |
| + drop the fake $50k rows | R² = 0.76 |
| + log-transform skewed features | R² = 0.79 |
| + interaction features (`RM × LSTAT`, etc.) | R² = 0.82 |
| **Gradient Boosting**, same features | **R² = 0.89** |

The biggest single lever was fixing the data, not the model. But once the data was clean, switching from Linear Regression to Gradient Boosting picked up the non-linear relationships (e.g. crime rate doesn't affect price in a straight line) that Linear Regression structurally can't capture — for free, with no extra feature engineering.

Final model: **Gradient Boosting, test R² ≈ 0.89, average error ≈ $2,400.**

## Results were checked properly, not just by R²

- Compared against a **baseline** (predicting the average price for every house) to confirm the model is actually learning something, not just producing plausible-looking numbers
- **RMSE reported as a % of average price** (~11%), not just a raw dollar figure
- **Predicted-vs-actual** and **residual plots** used to visually check for systematic errors, not just a single summary statistic

## The Excel dashboard

A business-facing view of the same analysis — KPI cards (average price, homes analyzed, top price driver, model accuracy), a color-coded correlation chart, and scatter plots with trendlines. Every number is a live formula (`CORREL`, `AVERAGEIF`, `AVERAGE`) referencing the raw data sheet, so it recalculates if the underlying data changes.

## The Streamlit app

A live, public tool where anyone can adjust sliders for a property's features (rooms, crime rate, tax rate, distance to job centers, etc.) and get an instant predicted price, along with a breakdown of which inputs drove that specific prediction.

Try it: **[https://bostonhouseprediction-xcvrykfektvw8acxuarrhq.streamlit.app/](https://bostonhousingpricepredictor-ppnmoe7bpabxn3h3uu7keu.streamlit.app)**

The app trains its Gradient Boosting model fresh on startup (cached after first load with `@st.cache_resource`) rather than loading a pickled model file — this avoids scikit-learn version mismatches between the local environment and the deployment server.

## Running it locally

```bash
cd Boston_House_Prediction
pip install -r requirements.txt
streamlit run app.py
```

Keep `boston.csv` in the same folder as `app.py` — it's loaded with a relative path.

## A note on the data

This is the classic Boston Housing dataset from a 1978 study. It includes a column (`B`) measuring the proportion of Black residents by town, originally included as a socioeconomic proxy. It's been widely criticized for encoding racial bias, and scikit-learn removed the dataset from its library in 2020 for this reason. It's used here as a well-known teaching dataset for practicing regression and deployment, not as an endorsement of the original methodology — the app surfaces this column only in an "advanced" panel, not as a primary input.

## Tech stack

Python · pandas · scikit-learn · matplotlib / seaborn · Streamlit · openpyxl
