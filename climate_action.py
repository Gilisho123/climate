# =============================================
# 🌱 Giltech Online Cyber – AI for Climate Action
# Predicting Carbon Emissions using Machine Learning
# =============================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load dataset
# For demonstration, we’ll use a sample dataset from the World Bank (simulated)
# Columns: Country, Year, GDP, Energy_Use, Population, CO2_Emissions
data = {
    "Country": ["Kenya"] * 10,
    "Year": list(range(2013, 2023)),
    "GDP": [55, 58, 61, 65, 70, 74, 79, 84, 89, 95],
    "Energy_Use": [45, 47, 50, 52, 55, 58, 60, 63, 67, 70],
    "Population": [44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "CO2_Emissions": [10.5, 10.9, 11.2, 11.8, 12.3, 12.9, 13.5, 14.1, 14.7, 15.2]
}

df = pd.DataFrame(data)

# Step 3: Explore data
print("Dataset preview:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Step 4: Prepare data for training
X = df[["GDP", "Energy_Use", "Population"]]
y = df["CO2_Emissions"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train models
lin_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lin_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Step 6: Predict
lin_pred = lin_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Step 7: Evaluate models
print("\nModel Performance:")
print("Linear Regression R²:", round(r2_score(y_test, lin_pred), 3))
print("Random Forest R²:", round(r2_score(y_test, rf_pred), 3))
print("Linear MAE:", round(mean_absolute_error(y_test, lin_pred), 3))
print("Random Forest MAE:", round(mean_absolute_error(y_test, rf_pred), 3))

# Step 8: Visualize results
plt.figure(figsize=(8,5))
plt.plot(df["Year"], df["CO2_Emissions"], label="Actual CO₂ Emissions", marker='o')
plt.plot(df["Year"].iloc[-len(rf_pred):], rf_pred, label="Predicted CO₂ (Random Forest)", linestyle='--', marker='x')
plt.title("CO₂ Emissions Prediction – Giltech Online Cyber (SDG 13)")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (tons per capita)")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Future Prediction (example)
future_data = pd.DataFrame({
    "GDP": [105, 115, 125],
    "Energy_Use": [75, 80, 85],
    "Population": [54, 55, 56]
})

future_pred = rf_model.predict(future_data)
print("\n🌍 Future CO₂ Emissions Predictions (Next 3 Years):")
for year, pred in zip([2023, 2024, 2025], future_pred):
    print(f"{year}: {pred:.2f} tons per capita")

# Step 10: Ethical Reflection
print("""
Ethical Reflection:
-------------------
✔ This model supports SDG 13 (Climate Action) by forecasting carbon emissions.
✔ Helps promote awareness among governments, businesses, and citizens.
✔ Data bias can occur due to missing records in developing countries.
✔ Giltech Online Cyber encourages responsible AI use for sustainability.
""")