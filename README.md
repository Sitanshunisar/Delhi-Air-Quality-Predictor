# Delhi-AQI-Flask-App

### Overview

This is a Flask app that predicts the PM2.5 based on the Delhi-AQI dataset. The app serves an interactive dashboard that visually compares the performance of three different machine learning models to analyze air quality.

### Model

This project compares the "inference" power of a static model against the "forecasting" power of two Recurrent Neural Network (RNN) sequence models. After engineering cyclical "hour" features (`hour_sin`, `hour_cos`) and running on a proper temporal-split test set, the results are:

* **1. StackingRegressor:** The "inference" model (using CatBoost, LightGBM, and Ridge). It predicts the 10 AM value using other 10 AM data, achieving a high **$R^2$ score of 0.9618**.

* **2. GRU (Gated Recurrent Unit):** The best *forecasting* model. It predicts the 10 AM value using *only* data from 9 AM and earlier. It achieved an **$R^2$ score of 0.8905**.

* **3. LSTM (Long Short-Term Memory):** The second forecasting model, which was outperformed by the simpler GRU on this dataset, achieving an **$R^2$ score of 0.8359**.

### Technologies Used

* Python
* Flask
* Pandas
* Numpy
* Scikit-learn
* TensorFlow (Keras)
* CatBoost
* LightGBM
* HTML/CSS
* Chart.js

### How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the models (Do this ONCE):**
    This script will create all 6 model files and place them in the `models/` folder.
    ```bash
    python notebooks/train_all_models.py
    ```
4.  Run the Flask app:
    ```bash
    python app.py
    ```
5.  Open your browser and go to `http://127.0.0.1:5000`