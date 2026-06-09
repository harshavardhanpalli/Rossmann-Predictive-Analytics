import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/processed/train_features.csv")

# Drop date column
df = df.drop(columns=["Date"])

# Convert categorical columns to numbers
df = pd.get_dummies(
    df,
    columns=[
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval"
    ],
    drop_first=True
)

# Target variable
y = df["Sales"]

# Features
X = df.drop("Sales", axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save model
joblib.dump(model, "models/random_forest.pkl")

print("Model saved successfully!")