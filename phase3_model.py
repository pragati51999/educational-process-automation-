import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Create 'models' folder if not exists
os.makedirs("models", exist_ok=True)

# Load cleaned dataset
data = pd.read_csv("data/student_data_cleaned.csv")

# Features and target
X = data[['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores']]
y = data['Exam_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R²:", r2_score(y_test, y_pred_lr))

# Save model
joblib.dump(lr_model, "models/linear_regression_model.pkl")

# Random Forest model (optional)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R²:", r2_score(y_test, y_pred_rf))

# Save model
joblib.dump(rf_model, "models/random_forest_model.pkl")

