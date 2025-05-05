# 🌧️ Rainfall Prediction Project (Germany)

## Full Project Overview

This project predicts rainfall levels in various German cities using advanced machine learning models, feature engineering, and interactive visualization tools — all wrapped inside a modern GUI application.

---

## 🧠 Machine Learning & Features

### 🔧 Feature Engineering:
- `Temperature²`  
- `Humidity²`  
- `Temp × Humidity`  

### 🤖 Models Trained:
- Linear Regression
- Random Forest
- Tuned Random Forest (via `GridSearchCV`)
- SVR (Support Vector Regression)

### 📈 Evaluation Metrics:
- MAE (Mean Absolute Error)  
- R² (Coefficient of Determination)  
- Model comparison table with all models included

---

## 📊 Visualizations

- **Model Comparison Table**:
  - Built using Plotly
  - Styled dark theme with interactive sorting
  - Includes all models with performance metrics

- **Scatter Plot**:
  - Actual vs. Predicted Rainfall (Tuned Random Forest)
  - Ideal prediction line for visual error analysis
  - Modern color scheme with hover and zoom

---

## 🗺️ Interactive Rainfall Map

- Built using Plotly or Folium
- Shows **average rainfall per city**
- Opens in browser with one click
- Clean marker styling and info tooltips

---

## 💻 Graphical User Interface (GUI)

### ⚙️ Built with:
- `Tkinter` and `PIL` for desktop GUI
- Responsive design layout

### Fully Redesigned Interface:
- **Dark mode** with themed background image
- **Blue hover effects** on buttons
- Modern spacing, fonts, and layout alignment
- Custom **emoji-based icons**
- Styled buttons:  
  - “Run Prediction” (shows results)  
  - “Show Rainfall Map” (opens map)

### 📈 From GUI:
- Automatically opens visual charts (table + scatter)
- Real-time prediction runs after button click

---

## 📄 Files & Structure

- `main.py` – GUI + ML logic  
- `Rainfall_Data_Germany_Complete.csv` – Cleaned dataset  
- `rainfall_map_germany.html` – Interactive map  
- `umbrella_icon.png`, `pin_icon.png`, etc. – GUI assets  
- `README.md` – Documentation  
- `rainfall_app_background.png` – Background image  

---

## 🛠️ Installation & Usage

```bash
pip install pandas numpy scikit-learn plotly Pillow
python main.py
