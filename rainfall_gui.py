import tkinter as tk
from PIL import Image, ImageTk
import webbrowser
import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# Run prediction and display results
def run_prediction():
    df = pd.read_csv("Rainfall_Data_Germany_Complete.csv")
    df = df.dropna()
    df.columns = df.columns.str.strip()
    df["Temperature (°C)"] = pd.to_numeric(df["Temperature (°C)"], errors="coerce")
    df["Humidity (%)"] = pd.to_numeric(df["Humidity (%)"], errors="coerce")
    df = df.dropna()
    df = pd.get_dummies(df, columns=["City", "Climate_Type"], drop_first=True)

    df["Temp^2"] = df["Temperature (°C)"] ** 2
    df["Humidity^2"] = df["Humidity (%)"] ** 2
    df["Temp*Humidity"] = df["Temperature (°C)"] * df["Humidity (%)"]

    features = ["Latitude", "Longitude", "Month", "Year", "Elevation (m)", "Temperature (°C)",
                "Humidity (%)", "Temp^2", "Humidity^2", "Temp*Humidity"]
    features += [col for col in df.columns if col.startswith("City_") or col.startswith("Climate_Type_")]
    X = df[features]
    y = df["Rainfall (mm)"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(x_train, y_train)
    rf = RandomForestRegressor(random_state=42).fit(x_train, y_train)
    svr = SVR().fit(x_train, y_train)

    pred_lr = lr.predict(x_test)
    pred_rf = rf.predict(x_test)
    pred_svr = svr.predict(x_test)

    mae_lr, r2_lr = mean_absolute_error(y_test, pred_lr), r2_score(y_test, pred_lr)
    mae_rf, r2_rf = mean_absolute_error(y_test, pred_rf), r2_score(y_test, pred_rf)
    mae_svr, r2_svr = mean_absolute_error(y_test, pred_svr), r2_score(y_test, pred_svr)

    # Tune RF
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2]
    }, cv=3, scoring='neg_mean_absolute_error')
    grid_search.fit(x_train, y_train)
    best_rf = grid_search.best_estimator_
    pred_best_rf = best_rf.predict(x_test)
    mae_best_rf, r2_best_rf = mean_absolute_error(y_test, pred_best_rf), r2_score(y_test, pred_best_rf)

        # Helper to force black text
    def html_black(text):
        return f"<span style='color:black'>{text}</span>"

    # TABLE
    table = go.Figure(data=[go.Table(
        columnwidth=[120, 100, 100],
        header=dict(
        values=["<b>Model</b>", "<b>MAE</b>", "<b>R²</b>"],
        fill=dict(color="#50bcd3"),
        line_color="black",
        align="center",
        font=dict(color="white", size=18, family="Segoe UI")),
        cells=dict(
        values=[
            ["Linear Regression", "Random Forest", "Tuned RF", "SVR"],
            [f"{mae_lr:.2f}", f"{mae_rf:.2f}", f"{mae_best_rf:.2f}", f"{mae_svr:.2f}"],
            [f"{r2_lr:.2f}", f"{r2_rf:.2f}", f"{r2_best_rf:.2f}", f"{r2_svr:.2f}"]
        ],
        fill=dict(color="#2b2b2b"),  # ← DARK GRAY for white text visibility
        line_color="black",
        align="center",
        font=dict(color="white", size=16, family="Segoe UI"),
        height=38
    )
)])
    table.update_layout(
    paper_bgcolor="black",
    plot_bgcolor="black",
    title="Model Comparison Table",
    title_font=dict(color="#50bcd3", size=22, family="Segoe UI")
)
    table.show()


    # SCATTER
    scatter = px.scatter(
        x=y_test, y=pred_best_rf,
        labels={"x": "Actual Rainfall (mm)", "y": "Predicted Rainfall (mm)"},
        title="Actual vs Predicted Rainfall (Tuned Random Forest)",
        template="plotly_dark"
    )
    scatter.add_trace(go.Scatter(
        x=y_test, y=y_test,
        mode="lines", name="Ideal Prediction",
        line=dict(dash="dash", color="#ffaa55")
    ))
    scatter.update_layout(
        title_font=dict(color="#00b3b3", size=22),
        font=dict(color="#00b3b3"),
        xaxis=dict(title_font=dict(size=16, color="#00b3b3")),
        yaxis=dict(title_font=dict(size=16, color="#00b3b3"))
    )
    scatter.show()

def open_map():
    try:
        webbrowser.open(f"file://{os.path.abspath('rainfall_map_germany.html')}")
    except Exception as e:
        print("Error opening map:", e)

# GUI Setup
root = tk.Tk()
root.title("Rainfall Prediction (Styled Final)")
root.geometry("1152x768")
root.resizable(False, False)

canvas = tk.Canvas(root, width=1152, height=768, highlightthickness=0)
canvas.pack()

bg_img = Image.open("rainfall_app_background.png").resize((1152, 768))
bg_photo = ImageTk.PhotoImage(bg_img)
canvas.create_image(0, 0, anchor="nw", image=bg_photo)

umbrella_icon = ImageTk.PhotoImage(Image.open("umbrella_icon.png").resize((120, 120)))
rain_icon = ImageTk.PhotoImage(Image.open("rain_icon.png").resize((32, 32)))
pin_icon = ImageTk.PhotoImage(Image.open("pin_icon.png").resize((32, 32)))

canvas.create_image(576, 90, image=umbrella_icon)
canvas.create_text(576, 165, text="Rainfall Prediction Results", fill="#50bcd3", font=("Segoe UI", 28, "bold"))

# Results
models = [
    ("Linear Regression", "→ MAE: 50.73 | R²: -0.00"),
    ("Random Forest",     "→ MAE: 52.30 | R²: -0.10"),
    ("Tuned RF",          "→ MAE: 51.82 | R²: -0.06")
]
for i, (model, value) in enumerate(models):
    y = 250 + i * 60
    canvas.create_image(340, y, image=rain_icon)
    canvas.create_text(390, y, text=model, fill="#ffaa55", anchor="w", font=("Segoe UI", 18, "bold"))
    canvas.create_text(650, y, text=value, fill="#aadce8", anchor="w", font=("Segoe UI", 18))

# Button hover
def on_enter(e): e.widget.config(bg="#002c3c", fg="#ffffff")
def on_leave(e): e.widget.config(bg="#0f0f0f", fg="#ffaa55")

# Create styled buttons
def create_button(x, y, icon, text, command):
    btn = tk.Button(root, text=" " + text, image=icon, compound="left",
                    font=("Segoe UI", 13, "bold"), fg="#ffaa55",
                    bg="#0f0f0f", activeforeground="#ffffff",
                    activebackground="#001a1a", relief="solid", bd=1,
                    highlightbackground="#50bcd3", highlightthickness=2, cursor="hand2")
    btn.place(x=x, y=y, width=240, height=48)
    btn.config(command=command)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

# Buttons
create_button(360, 500, rain_icon, "Run Prediction", run_prediction)
create_button(630, 500, pin_icon, "Show Rainfall Map", open_map)

root.mainloop()
