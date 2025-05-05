import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess data
df = pd.read_csv("Rainfall_Data_Germany_Complete.csv")
df = df.dropna()
df.columns = df.columns.str.strip()
df["Temperature (°C)"] = pd.to_numeric(df["Temperature (°C)"], errors="coerce")
df["Humidity (%)"] = pd.to_numeric(df["Humidity (%)"], errors="coerce")
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, columns=["City", "Climate_Type"], drop_first=True)

# Feature engineering
df["Temp^2"] = df["Temperature (°C)"] ** 2
df["Humidity^2"] = df["Humidity (%)"] ** 2
df["Temp*Humidity"] = df["Temperature (°C)"] * df["Humidity (%)"]

# Features and target
features = ["Latitude", "Longitude", "Month", "Year", "Elevation (m)", "Temperature (°C)",
            "Humidity (%)", "Temp^2", "Humidity^2", "Temp*Humidity"]
features += [col for col in df.columns if col.startswith("City_") or col.startswith("Climate_Type_")]
X = df[features]
y = df["Rainfall (mm)"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr = LinearRegression().fit(x_train, y_train)
rf = RandomForestRegressor(random_state=42).fit(x_train, y_train)
svr = SVR().fit(x_train, y_train)

# Predictions
pred_lr = lr.predict(x_test)
pred_rf = rf.predict(x_test)
pred_svr = svr.predict(x_test)

# Metrics
mae_lr, r2_lr = mean_absolute_error(y_test, pred_lr), r2_score(y_test, pred_lr)
mae_rf, r2_rf = mean_absolute_error(y_test, pred_rf), r2_score(y_test, pred_rf)
mae_svr, r2_svr = mean_absolute_error(y_test, pred_svr), r2_score(y_test, pred_svr)

# GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(x_train, y_train)
best_rf = grid_search.best_estimator_

# Predictions with best RF model
pred_best_rf = best_rf.predict(x_test)
mae_best_rf = mean_absolute_error(y_test, pred_best_rf)
r2_best_rf = r2_score(y_test, pred_best_rf)

# Model comparison table with SVR
comparison_table = go.Figure(data=[go.Table(
    columnwidth=[120, 100, 100],
    header=dict(
        values=["<b>Model</b>", "<b>MAE</b>", "<b>R²</b>"],
        fill_color="#50bcd3",
        align="center",
        font=dict(color="white", size=18, family="Segoe UI")
    ),
    cells=dict(
        values=[
            ["Linear Regression", "Random Forest", "Tuned RF", "SVR"],
            [f"{mae_lr:.2f}", f"{mae_rf:.2f}", f"{mae_best_rf:.2f}", f"{mae_svr:.2f}"],
            [f"{r2_lr:.2f}", f"{r2_rf:.2f}", f"{r2_best_rf:.2f}", f"{r2_svr:.2f}"]
        ],
        fill_color="lightgray",
        align="center",
        font=dict(color="black", size=16, family="Segoe UI")
    )
)])
comparison_table.update_layout(
    title="Model Comparison Table",
    title_font=dict(color="#50bcd3", size=22, family="Segoe UI"),
    paper_bgcolor="white"
)
comparison_table.show()

# Scatter: Actual vs Predicted (Tuned RF)
scatter_fig = px.scatter(
    x=y_test,
    y=pred_best_rf,
    labels={"x": "Actual Rainfall (mm)", "y": "Predicted Rainfall (mm)"},
    title="Actual vs Predicted Rainfall (Tuned Random Forest)"
)
scatter_fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Prediction Line',
                                 line=dict(color="#ffaa55", dash="dash")))
scatter_fig.update_layout(template="plotly_dark")
scatter_fig.show()
