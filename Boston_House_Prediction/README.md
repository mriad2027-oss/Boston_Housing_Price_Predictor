# Boston Housing Price Predictor — Streamlit App

A live web app that predicts a home's market value from property and neighborhood features, using a Gradient Boosting model trained on the classic Boston Housing dataset (test R² ≈ 0.89).

## Try it locally

```bash
pip install -r requirements.txt
python train_model.py     # trains the model and saves model.pkl
streamlit run app.py
```

## Files

- `app.py` — the Streamlit app
- `train_model.py` — trains the Gradient Boosting model and saves `model.pkl` + `feature_names.pkl`
- `boston.csv` — the dataset
- `model.pkl`, `feature_names.pkl` — the trained model, already generated (no need to retrain unless you want to)

## Deploying it publicly (free)

1. Push this whole folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click "New app", pick your repo, set the main file to `app.py`
4. Deploy — you'll get a public URL like `yourapp.streamlit.app` that anyone can open, no login needed

## A note on the data

This dataset includes a column (`B`) measuring the proportion of Black residents by town, from the original 1970s study. It's been widely criticized for encoding racial bias, and scikit-learn removed the dataset from its library in 2020 for this reason. It's included here only for completeness (tucked into the app's "advanced" panel) and used for learning purposes, not as an endorsement of the original methodology.
