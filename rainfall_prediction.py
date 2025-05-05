# 1. Import libraries
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Load dataset
df = pd.read_csv("Rainfall_Data_Germany_Complete.csv")
df.columns = df.columns.str.strip()  # Clean column names

# 3. Drop rows with missing values in key columns
df = df.dropna(subset=["temp", "humidity", "dew", "precip", "datetime", "windspeed", "cloudcover"])

# 4. Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])  # Drop rows with invalid datetime
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

# 5. Encode categorical feature: preciptype (e.g. rain, snow)
df["preciptype"] = df["preciptype"].fillna("")
df["precip_rain"] = df["preciptype"].apply(lambda x: 1 if 'rain' in x.lower() else 0)
df["precip_snow"] = df["preciptype"].apply(lambda x: 1 if 'snow' in x.lower() else 0)

# 6. Define features and target
features = [
    "temp", "humidity", "dew", "windspeed", "cloudcover",
    "month", "year", "precip_rain", "precip_snow"
]
X = df[features]
y = df["precip"]  # Target: rainfall in mm

# 7. Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR()
}

results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append((name, mae, r2))

# 9. Show comparison table
comparison = go.Figure(data=[go.Table(
    header=dict(values=["Model", "MAE", "R²"]),
    cells=dict(values=[[r[0] for r in results],
                       [f"{r[1]:.2f}" for r in results],
                       [f"{r[2]:.2f}" for r in results]])
)])
comparison.update_layout(title="Model Performance Comparison")
comparison.show()

# 10. Actual vs Predicted plot for best model (Random Forest)
best_model = models["Random Forest"]
y_pred_best = best_model.predict(x_test)

scatter = px.scatter(
    x=y_test, y=y_pred_best,
    labels={"x": "Actual Precipitation (mm)", "y": "Predicted Precipitation (mm)"},
    title="Random Forest: Actual vs Predicted Rainfall"
)
scatter.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit',
                             line=dict(color="orange", dash="dash")))
scatter.update_layout(template="plotly_white")
scatter.show()
