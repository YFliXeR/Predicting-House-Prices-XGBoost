import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

data = fetch_california_housing()
data.feature_names # Use print() to see the column names
data.DESCR # Use print() to see a detailed description of the dataset 

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# XGBoost model
n_estimators = 100
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# Fitting the model and measuring time
start_train_time = time.time()
xgb.fit(X_train, y_train)
end_train_time = time.time()
xgb_train_time = end_train_time - start_train_time

# Prediction and measuring time
start_predict_time = time.time()
prediction = xgb.predict(X_test)
end_predict_time = time.time()
xgb_predict_time = end_predict_time - start_predict_time

# Evaluation
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"Model MSE = {mse:.4f}")
print(f"Model R^2 = {r2:.4f}")
# MSE (Mean Squared Error): measures how far off the predictions are on average (lower is better)
# R² (R-squared): shows how well the model explains the variation in house prices (closer to 1 is better)
print(f"Model Training Time  = {xgb_train_time:.4f} seconds")
print(f"Model Prediction Time = {xgb_predict_time:.4f} seconds")

# Visualization 
# Converting values to actual dollars
y_actual = y_test * 100000
prediction_actual_dollars = prediction * 100000
plt.scatter(y_actual, prediction_actual_dollars, alpha=0.5, label="Model Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
min_val = min(y_actual.min(), prediction_actual_dollars.min())
max_val = max(y_actual.max(), prediction_actual_dollars.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect model")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()