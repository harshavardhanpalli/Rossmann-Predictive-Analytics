import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest.pkl")

# Load dataset
df = pd.read_csv("data/processed/train_features.csv")

# Drop Date
df = df.drop(columns=["Date"])

# Encode categories
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

# Features only
X = df.drop("Sales", axis=1)

# Last record
future_data = X.tail(30)

# Forecast
forecast = model.predict(future_data)

print("Next 30 Predictions")
print(forecast)