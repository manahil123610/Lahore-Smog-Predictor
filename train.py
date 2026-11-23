import pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("lahore_daily.csv", index_col=0, parse_dates=True)

# Features: today's values + last 3 days of PM2.5 + month
for k in (1, 2, 3):
    df[f"pm2_5_lag{k}"] = df["pm2_5"].shift(k)
df["month"] = df.index.month
df["target"] = df["pm2_5"].shift(-1)      # tomorrow's PM2.5
df = df.dropna()

FEATURES = ["pm2_5", "pm10", "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "surface_pressure",
            "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", "month"]

# Time-based split: last 20% of days = test
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
Xtr, ytr = train[FEATURES], train["target"]
Xte, yte = test[FEATURES], test["target"]

def report(name, pred):
    print(f"{name:30s} MAE={mean_absolute_error(yte, pred):6.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(yte, pred)):6.2f}  R2={r2_score(yte, pred):.3f}")

lr = LinearRegression().fit(Xtr, ytr)
rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3,
                           random_state=42, n_jobs=-1).fit(Xtr, ytr)
preds = {"Persistence (tomorrow=today)": test["pm2_5"].values,
         "Linear Regression": lr.predict(Xte),
         "Random Forest": rf.predict(Xte)}
for k, v in preds.items():
    report(k, v)

# Charts
plt.figure(figsize=(11, 4))
plt.plot(test.index, yte, label="Actual", color="#3A1C2B")
plt.plot(test.index, preds["Random Forest"], label="Predicted", color="#E8447A", alpha=.8)
plt.ylabel("PM2.5 (µg/m³)"); plt.title("Next-day PM2.5: actual vs predicted (test set)")
plt.legend(); plt.tight_layout(); plt.savefig("pred_vs_actual.png", dpi=150)

plt.figure(figsize=(7, 4))
pd.Series(rf.feature_importances_, index=FEATURES).sort_values().plot.barh(color="#E8447A")
plt.title("Feature importance"); plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150)

plt.figure(figsize=(11, 4))
df["pm2_5"].resample("MS").mean().plot(color="#3A1C2B", title="Monthly average PM2.5 in Lahore")
plt.ylabel("µg/m³"); plt.tight_layout(); plt.savefig("monthly_pm25.png", dpi=150)

joblib.dump({"model": rf, "features": FEATURES}, "model.joblib", compress=3)
