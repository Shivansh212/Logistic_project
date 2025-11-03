# End-to-End Delivery Time Prediction

This project is a complete end-to-end Machine Learning application that predicts the time taken for a delivery.

The entire ML pipeline is built from scratch, including data ingestion, feature engineering, data transformation, model training, and evaluation. The best-performing model is saved and exposed via a simple web application (built with Flask/FastAPI) where a user can input new data and get an instant prediction.

**[Live Demo Link](https://logistic-project.onrender.com)** *(Note: This appears to be a development tunnel and may not be active permanently.)*

---

## 🚀 Features

* **Data Ingestion:** Reads the raw dataset (`amazon_delivery.csv`).
* **Feature Engineering:**
    * Calculates `Haversine_distance` from latitude and longitude.
    * Extracts detailed features from `Order_Date` and `Order_Time` (Year, Month, Day, Hour, Minute, Day of Week).
    * Calculates `Total_preparation_time` by finding the difference between order and pickup times.
    * Converts the target variable `Delivery_Time` into hours.
* **Data Transformation:**
    * Builds a `ColumnTransformer` preprocessing pipeline.
    * **Numerical Features:** Imputes missing values with `median` and scales using `RobustScaler` (which is robust to outliers).
    * **Categorical Features:** Imputes missing values with `most_frequent`, applies `OneHotEncoder`, and scales using `RobustScaler`.
* **Model Training & Selection:**
    * Trains 10 different regression models:
        * Linear Regression
        * Lasso
        * Ridge
        * K-Neighbors Regressor
        * Decision Tree
        * Random Forest
        * XGBRegressor
        * CatBoost Regressor
        * AdaBoost Regressor
        * Gradient Boosting
    * Uses `GridSearchCV` to find the best hyperparameters for key models.
    * Evaluates all models based on their **R² score** and **Mean Absolute Error (MAE)**.
* **Model Deployment:**
    * Saves the best-performing model (`best_model.pkl`) and the preprocessor (`preprocessor.pkl`).
    * A prediction pipeline (`prediction_pipeline.py`) loads these artifacts to make predictions on new data.
    * A web server (`app.py`, not provided but inferred) serves a UI to interact with the model.

