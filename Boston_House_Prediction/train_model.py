import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(APP_DIR, "boston.csv"))

# same cleanup as the notebook - drop the fake capped prices
df = df[df["MEDV"] < 50].copy()

# log the skewed columns
df["CRIM"] = np.log1p(df["CRIM"])
df["DIS"] = np.log1p(df["DIS"])
df["LSTAT"] = np.log1p(df["LSTAT"])

# interaction features
df["RM_LSTAT"] = df["RM"] * df["LSTAT"]
df["TAX_per_room"] = df["TAX"] / df["RM"]
df["DIS_RAD"] = df["DIS"] / df["RAD"]

x = df.drop("MEDV", axis=1)
y = df["MEDV"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print("Test R2:", r2_score(y_test, pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

joblib.dump(model, os.path.join(APP_DIR, "model.pkl"))
joblib.dump(list(x.columns), os.path.join(APP_DIR, "feature_names.pkl"))
print("saved model.pkl and feature_names.pkl")
