import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/train_features.csv")

# Drop date
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

# Load model
model = joblib.load("models/random_forest.pkl")

# Features
X = df.drop("Sales", axis=1)

# Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(15))
importance.to_csv(
    "reports/insights/feature_importance.csv",
    index=False
)
# Plot
plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"].head(15),
    importance["Importance"].head(15)
)

plt.title("Top 15 Important Features")
plt.tight_layout()

plt.show()