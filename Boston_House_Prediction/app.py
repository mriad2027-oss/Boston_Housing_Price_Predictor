import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Boston Housing Price Predictor", page_icon="🏠", layout="wide")

model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🏠 Boston Housing Price Predictor")
st.write("Adjust the sliders to describe a property and neighborhood, and get a predicted market value.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Property")
    rm = st.slider("Average rooms per dwelling", 3.0, 9.0, 6.0, 0.1)
    age = st.slider("% of units built before 1940", 0.0, 100.0, 50.0, 1.0)
    tax = st.slider("Property tax rate (per $10,000)", 180, 720, 300, 5)

with col2:
    st.subheader("Neighborhood")
    crim = st.slider("Crime rate (per capita)", 0.0, 90.0, 3.0, 0.1)
    lstat = st.slider("% lower status population", 1.0, 40.0, 12.0, 0.5)
    ptratio = st.slider("Pupil-teacher ratio", 12.0, 22.0, 18.0, 0.1)

with col3:
    st.subheader("Location")
    dis = st.slider("Distance to employment centers", 1.0, 13.0, 4.0, 0.1)
    rad = st.slider("Highway accessibility index", 1, 24, 5, 1)
    chas = st.checkbox("Borders the Charles River")

with st.expander("Advanced / less common features"):
    zn = st.slider("% land zoned for large lots", 0.0, 100.0, 10.0, 1.0)
    indus = st.slider("% non-retail business acres", 0.0, 30.0, 10.0, 0.5)
    nox = st.slider("Nitric oxide concentration", 0.35, 0.90, 0.55, 0.01)
    b_val = st.slider("B (demographic composition index)", 0.0, 400.0, 390.0, 1.0)

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):

    crim_log = np.log1p(crim)
    dis_log = np.log1p(dis)
    lstat_log = np.log1p(lstat)

    row = {
        "CRIM": crim_log,
        "ZN": zn,
        "INDUS": indus,
        "CHAS": int(chas),
        "NOX": nox,
        "RM": rm,
        "AGE": age,
        "DIS": dis_log,
        "RAD": rad,
        "TAX": tax,
        "PTRATIO": ptratio,
        "B": b_val,
        "LSTAT": lstat_log,
        "RM_LSTAT": rm * lstat_log,
        "TAX_per_room": tax / rm,
        "DIS_RAD": dis_log / rad,
    }

    x_input = pd.DataFrame([row])[feature_names]
    prediction = model.predict(x_input)[0]
    predicted_price = prediction * 1000

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        st.metric("Predicted Home Value", f"${predicted_price:,.0f}")
        st.caption("Model: Gradient Boosting, test R² ≈ 0.89, average error ≈ $2,400")

    with result_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_price,
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [0, 50000]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 15000], "color": "#fee2e2"},
                    {"range": [15000, 30000], "color": "#fef9c3"},
                    {"range": [30000, 50000], "color": "#dcfce7"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What drove this prediction")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(8)
    st.bar_chart(importances)

st.divider()
st.caption(
    "Trained on the classic Boston Housing dataset. This dataset includes a demographic "
    "composition feature (B) from the original 1970s study that has been criticized for "
    "encoding racial bias — kept here for historical accuracy, shown only in the advanced panel."
)
