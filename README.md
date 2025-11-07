# 🌍 Delhi PM2.5 Air Quality Prediction 🌬️

<p align="center">
  <img src="https://img.shields.io/badge/R%C2%B2%20Score-0.9907-brightgreen" alt="R² Score Badge">
  <img src="https://img.shields.io/badge/Model-Stacking%20Regressor-blue" alt="Model Badge">
  <img src="https://img.shields.io/badge/Status-Complete-success" alt="Project Status">
</p>

## 📖 Table of Contents
* [Problem Statement](#-problem-statement)
* [Methodology](#-methodology)
* [Model Performance](#-model-performance)
* [How to Use the Model](#-how-to-use-the-pre-trained-model)
* [Getting Started](#-getting-started)
* [Technology Stack](#-technology-stack)
* [Future Work](#-future-work)

---

## 🎯 Problem Statement

Air pollution is a critical environmental issue in metropolitan areas like Delhi. **Particulate Matter 2.5 (PM2.5)** is a key pollutant that poses significant health risks. Accurate and timely prediction of PM2.5 levels is essential for public health warnings and environmental policy-making.

This project aims to develop a robust machine learning model that accurately predicts PM2.5 concentrations based on the levels of other atmospheric pollutants.

---

## 🔬 Methodology

The project follows a structured data science workflow, from data exploration to model deployment.

### 1. Exploratory Data Analysis (EDA) & Feature Engineering
The initial analysis revealed strong correlations between several pollutants, particularly a **0.95 correlation between CO and PM2.5**. This insight guided the creation of new, informative features to enhance the model's learning capabilities. Engineered features include:
- **Pollutant Ratios**: e.g., `co_to_pm2_5`, `pm2_5_to_pm10`.
- **Interaction Terms**: e.g., `so2 * o3`.
- **Aggregate Metrics**: e.g., `total_pollutant`, `nox` (NO + NO2).

### 2. Dimensionality Reduction with PCA
To address multicollinearity among the highly correlated features and to reduce model complexity, **Principal Component Analysis (PCA)** was applied. The feature set was transformed into 14 principal components, which collectively capture the most significant variance in the data, leading to a more efficient and stable model.

### 3. High-Performance Ensemble Modeling
A **Stacking Regressor** was chosen to leverage the strengths of multiple diverse models.
- **Base Models**: `CatBoost Regressor` and `LGBM Regressor`, both powerful gradient boosting algorithms.
- **Meta-Model**: `Ridge Regression` was used as the final estimator to combine the predictions from the base models, adding L2 regularization to prevent overfitting and improve generalization.

---

## 🏆 Model Performance

The final stacked model demonstrates outstanding predictive accuracy, explaining over 99% of the variance in PM2.5 levels.

* **Root Mean Squared Error (RMSE)**: **18.46**
* **Coefficient of Determination (R² Score)**: **0.9907**

The scatter plot below visualizes the model's precision, showing a nearly perfect alignment between the actual and predicted values.

<p align="center">
  <em>Actual vs. Predicted PM2.5 Concentrations</em>
  <br>
  <img src="outputs/graphs/actual_vs_predicted_scatter.png" width="600">
  <br>
  <em>Note: This graph is generated in the <code>outputs/graphs</code> directory upon running the notebook.</em>
</p>

---

## 💡 How to Use the Pre-trained Model

The `models/` directory contains the saved `scaler`, `pca` object, and the final `stacked_regressor_model`. Here’s how you can use them to make a new prediction:

```python
import joblib
import pandas as pd
import numpy as np

# Load the saved components
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca.pkl')
model = joblib.load('models/stacked_regressor_model.pkl')

# Create a sample DataFrame with new data (must have all original columns except the target)
# The order of columns must match the training data
new_data = pd.DataFrame({
    'co': [1800.2], 'no': [5.8], 'no2': [41.2], 'o3': [10.1], 'so2': [20.5],
    'pm10': [200.1], 'nh3': [6.5],
    # Add all engineered features as well
    'co_to_pm2_5': [10.0], 'no_to_pm2_5': [0.03], 'nox': [47.0],
    'nox_to_pm2_5': [0.26], 'total_pollutant': [2261.1], 'avg_pollutant': [282.6],
    'pm2_5_to_pm10': [0.9], 'co_no_interaction': [10441.16], 'so2_o3_interaction': [207.05],
    'co_squared': [3240720.04], 'log_pm2_5': [5.19], 'high_pm2_5': [1], 'high_co': [0]
})

# Apply the transformations
scaled_data = scaler.transform(new_data)
pca_data = pca.transform(scaled_data)

# Make a prediction
prediction = model.predict(pca_data)

print(f"Predicted PM2.5 Concentration: {prediction[0]:.2f}")