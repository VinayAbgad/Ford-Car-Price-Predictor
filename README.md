# 🚗 Ford Car Price Predictor

A machine learning web application that predicts the estimated selling price of a Ford car based on its specifications.

## 📌 Project Overview

This project uses a Linear Regression model to predict Ford car prices.

The application is built using Streamlit and allows users to enter:

- Car Model
- Manufacturing Year
- Mileage
- Transmission
- Fuel Type
- Road Tax
- MPG
- Engine Size

The application then processes the input data and predicts the estimated selling price.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## ⚙️ Machine Learning Process

The following preprocessing steps are used:

1. One-Hot Encoding for categorical features
2. Standard Scaling for numerical features
3. Linear Regression for price prediction

## 📁 Project Files

```text
app.py                 # Streamlit application
requirements.txt       # Required Python libraries
LR_ford_car.pkl        # Trained Linear Regression model
scaler.pkl             # Saved StandardScaler
columns.pkl            # Encoded feature columns
README.md              # Project documentation
