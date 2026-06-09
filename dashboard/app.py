import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Rossmann Predictive Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/train_features.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")

df = load_data()
model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("📈 Rossmann Predictive Analytics Dashboard")
st.markdown(
    """
    End-to-End Retail Sales Forecasting using Machine Learning

    *Model:* Random Forest Regressor  
    *Dataset:* Rossmann Store Sales  
    *R² Score:* 0.9363
    """
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Dashboard Filters")

store_list = sorted(df["Store"].unique())

selected_store = st.sidebar.selectbox(
    "Select Store",
    store_list
)

filtered_df = df[df["Store"] == selected_store]

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader(f"🏪 Store {selected_store} Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    len(filtered_df)
)

col2.metric(
    "Average Sales",
    f"{filtered_df['Sales'].mean():,.0f}"
)

col3.metric(
    "Maximum Sales",
    f"{filtered_df['Sales'].max():,.0f}"
)

col4.metric(
    "Average Customers",
    f"{filtered_df['Customers'].mean():,.0f}"
)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

# -----------------------------
# SALES TREND
# -----------------------------
st.subheader("📊 Sales Trend")

sales_chart = filtered_df.copy()

if "Date" in sales_chart.columns:
    sales_chart["Date"] = pd.to_datetime(
        sales_chart["Date"]
    )

    sales_chart = sales_chart.sort_values(
        "Date"
    )

    st.line_chart(
        sales_chart.set_index("Date")["Sales"]
    )

# -----------------------------
# SALES DISTRIBUTION
# -----------------------------
st.subheader("📈 Sales Distribution")

st.bar_chart(
    filtered_df["Sales"].head(100)
)

# -----------------------------
# FORECAST SECTION
# -----------------------------
st.subheader("🔮 Forecast Sample")

model_df = df.copy()

if "Date" in model_df.columns:
    model_df = model_df.drop(columns=["Date"])

model_df = pd.get_dummies(
    model_df,
    columns=[
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval"
    ],
    drop_first=True
)

X = model_df.drop("Sales", axis=1)

sample = X.tail(30)

forecast = model.predict(sample)

forecast_df = pd.DataFrame({
    "Forecast_Day": range(1, 31),
    "Predicted_Sales": forecast
})

st.dataframe(
    forecast_df,
    use_container_width=True
)

# -----------------------------
# FORECAST CHART
# -----------------------------
st.subheader("📉 Next 30 Predictions")

st.line_chart(
    forecast_df.set_index(
        "Forecast_Day"
    )
)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("⭐ Feature Importance")

try:
    importance_df = pd.read_csv(
        "reports/insights/feature_importance.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    top_features = importance_df.head(10)

    ax.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    ax.set_title(
        "Top 10 Important Features"
    )

    st.pyplot(fig)

except:
    st.warning(
        "feature_importance.csv not found. Run evaluate.py first."
    )

# -----------------------------
# DOWNLOAD REPORT
# -----------------------------
st.subheader("⬇️ Download Forecast")

csv = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="forecast_report.csv",
    mime="text/csv"
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    """
    *Developed By:* Harsha Vardhan

    Predictive Analytics Using Historical Data | Data Science Internship Project
    """
)