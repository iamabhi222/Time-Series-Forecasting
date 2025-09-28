import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats.mstats import winsorize
import warnings

# Suppress harmless warnings from Prophet
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------- 1. Load and Prepare Data --------------------
try:
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: 'data.json' not found. Please make sure the file is in the same directory.")
    exit()


df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y")
df.sort_values("date", inplace=True)
df.rename(columns={"date": "ds", "total_debit": "y"}, inplace=True)

# -------------------- 2. Outlier Handling & Transformation --------------------
# Winsorization is a robust way to handle outliers by capping them instead of removing data.
df["y_winsorized"] = winsorize(df["y"], limits=[0.05, 0.05])
# Log transform helps stabilize variance and makes patterns easier for the models to learn.
df["y_log"] = np.log1p(df["y_winsorized"])

# -------------------- 3. Feature Engineering --------------------
# Create time-based features to make the regression model more powerful.
df['day_of_week'] = df['ds'].dt.dayofweek  # Monday=0, Sunday=6
df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int) # 1 if weekend, 0 otherwise
df['month'] = df['ds'].dt.month
df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
df["day_index"] = (df["ds"] - df["ds"].min()).dt.days

# -------------------- 4. Model Evaluation with Train-Test Split --------------------
# We'll hold out the last 10 days to test how well our models perform on unseen data.
train_df = df.iloc[:-10]
test_df = df.iloc[-10:]

print("--- Model Evaluation on Last 10 Days ---")

# --- Prophet Model Evaluation ---
prophet_model = Prophet(changepoint_prior_scale=0.1, seasonality_mode="multiplicative")
prophet_model.fit(train_df[['ds', 'y_log']].rename(columns={'y_log': 'y'}))
test_forecast_prophet = prophet_model.predict(test_df[['ds']])
# Invert the log transformation to compare with original scale
prophet_preds = np.expm1(test_forecast_prophet['yhat'])
prophet_mae = mean_absolute_error(test_df['y_winsorized'], prophet_preds)
print(f"Prophet Model MAE on Test Set:      ₹{prophet_mae:.2f}")

# --- Enhanced Regression Model Evaluation ---
features = ['day_index', 'day_of_week', 'is_weekend', 'month', 'week_of_year']
X_train, y_train = train_df[features], train_df['y_log']
X_test, y_test = test_df[features], test_df['y_log']

reg_model = HuberRegressor()
reg_model.fit(X_train, y_train)
reg_predictions_log = reg_model.predict(X_test)
# Invert the log transformation
reg_preds = np.expm1(reg_predictions_log)
reg_mae = mean_absolute_error(test_df['y_winsorized'], reg_preds)
print(f"Huber Regression MAE on Test Set: ₹{reg_mae:.2f}")

# --- Ensemble Evaluation ---
ensemble_preds = (prophet_preds.values + reg_preds) / 2
ensemble_mae = mean_absolute_error(test_df['y_winsorized'], ensemble_preds)
print(f"Ensemble Model MAE on Test Set:     ₹{ensemble_mae:.2f}\n")


# -------------------- 5. Final Forecasting for the Future --------------------
# Now, retrain models on ALL data to make the most accurate future predictions.
print("--- Retraining on Full Dataset for Final Forecast ---\n")

# --- Retrain Prophet on all data ---
prophet_full = Prophet(changepoint_prior_scale=0.1, seasonality_mode="multiplicative")
prophet_full.fit(df[['ds', 'y_log']].rename(columns={'y_log': 'y'}))
future_dates_df = prophet_full.make_future_dataframe(periods=5)
forecast_prophet_full = prophet_full.predict(future_dates_df)
prophet_future_forecast = forecast_prophet_full[['ds', 'yhat']].tail(5)

# --- Retrain Regression on all data ---
X_full, y_full = df[features], df['y_log']
reg_full = HuberRegressor().fit(X_full, y_full)

# Create future features for the regression model
future_reg_features = future_dates_df.tail(5).copy()
future_reg_features['day_of_week'] = future_reg_features['ds'].dt.dayofweek
future_reg_features['is_weekend'] = (future_reg_features['ds'].dt.dayofweek >= 5).astype(int)
future_reg_features['month'] = future_reg_features['ds'].dt.month
future_reg_features['week_of_year'] = future_reg_features['ds'].dt.isocalendar().week.astype(int)
future_reg_features["day_index"] = (future_reg_features["ds"] - df["ds"].min()).dt.days

X_future = future_reg_features[features]
reg_future_preds_log = reg_full.predict(X_future)

# --- Final Ensemble Forecast ---
ensemble_df = prophet_future_forecast.copy()
ensemble_df['reg_pred_log'] = reg_future_preds_log

# Combine the log-space predictions and then transform back
ensemble_df['ensemble_pred_log'] = (ensemble_df["yhat"] + ensemble_df['reg_pred_log']) / 2
ensemble_df["final_prediction"] = np.expm1(ensemble_df['ensemble_pred_log'])

total_recharge = ensemble_df['final_prediction'].sum()

print("✅ Ensemble Forecast (Next 5 Days):")
print(ensemble_df[['ds', 'final_prediction']].round(2).to_string(index=False))
print("-----------------------------------------")
print(f"💰 Recommended recharge for the next 5 days: ₹{total_recharge:.2f}")
print("-----------------------------------------\n")


# -------------------- 6. Visualization --------------------
plt.figure(figsize=(15, 7))

# Plot historical data
plt.plot(df['ds'], df['y_winsorized'], 'o-', label='Historical Daily Spend (Winsorized)', color='skyblue', markersize=4)

# Plot the final 5-day ensemble forecast
plt.plot(ensemble_df['ds'], ensemble_df['final_prediction'], 'ks-', label='Ensemble Forecast (Next 5 Days)', markersize=8, linewidth=2)

plt.title('Student Mess Spend: Historical Data and 5-Day Forecast', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount (₹)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()