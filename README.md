# Syntecxhub - House Price Prediction (Linear Regression)

This repository contains **Project 1: House Price Prediction**, developed as part of the Syntecxhub Machine Learning Internship Program (Week 1).

The project implements a **Linear Regression** model using `scikit-learn` to predict housing values in the Boston area.

---

## 📌 Project Overview
The goal of this project is to analyze housing features in the Boston area, explore their relationships, and train a supervised machine learning model to predict prices.

### Key Highlights:
1. **Dynamic Data Fetching**: Fetches the Boston Housing dataset directly from the CMU data repository, resolving the deprecated `load_boston` scikit-learn dependency.
2. **Exploratory Data Analysis (EDA)**: Visualizes feature correlations and scatter plots (e.g., number of rooms vs price).
3. **Evaluation Metrics**: Uses RMSE (Root Mean Squared Error) and $R^2$ score for model validation.
4. **Model Export**: Serializes the trained model into a pickle file (`linear_regression_model.pkl`) for future inference/deployment.

---

## 📊 Dataset Features
The dataset contains **506 instances** and **13 features**:
* `RM`: Average number of rooms per dwelling (Strong positive correlation with price)
* `LSTAT`: % lower status of the population (Strong negative correlation with price)
* `NOX`: Nitric oxides concentration (parts per 10 million)
* `CRIM`: Per capita crime rate by town
* `price` (Target): Median value of owner-occupied homes in $1000s

---

## 📈 Model Performance
The model was trained on an 80/20 split and evaluated with the following results:

| Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **RMSE** | `4.791` | `4.301` |
| **R² Score** | `0.729` | `0.779` |

* **Coefficient Insights**:
  * Each additional room (`RM`) adds **+$3,632.88** to the home's value.
  * Nitric oxides air pollution (`NOX`) has the highest negative coefficient impact.

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python installed along with the following packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Run the Notebook
Open and run all cells in the Jupyter notebook:
[House_Price_Prediction.ipynb](House_Price_Prediction.ipynb)

### 3. Load Saved Model
You can load the trained model in any python script to make quick predictions:
```python
import pickle
import numpy as np

# Load model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example house prediction
sample_features = np.array([[0.006, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]])
print(f"Predicted price: ${model.predict(sample_features)[0]:.2f} thousand")
```
