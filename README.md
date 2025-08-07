**Employee Performance Forecasting Using KPIs**

This project demonstrates how to forecast employee performance using key performance indicators (KPIs) through machine learning models. It includes both a regression model using Random Forest and a time-series model using LSTM for individual forecasting over time.
🔍 Features

    ✅ Random Forest Regression for performance prediction across multiple employees

    📈 LSTM Time-Series Forecasting for predicting future performance trends per employee

    🧪 Synthetic KPI dataset generation (simulates realistic HR data)

    🧠 Model training and evaluation with performance metrics

    📊 Optional Streamlit dashboard interface for interactive forecasting

    💾 Model serialization using joblib

📁 Project Files
File	Description
train_model.py	Trains a Random Forest model on synthetic KPI data
app.py	Streamlit app for forecasting performance scores
synthetic_employee_kpis.csv	Generated dataset of 100 employees and KPIs
employee_performance_model.pkl	Trained Random Forest model file
lstm_employee_forecast.py	LSTM-based time-series forecast for single employee
README.md	This documentation file
🛠️ How to Use
1. Install Required Libraries

pip install pandas numpy scikit-learn streamlit matplotlib tensorflow

2. Train Random Forest Model

python train_model.py

3. Launch Streamlit App (Optional)

streamlit run app.py

4. Run LSTM Time Series Forecast

python lstm_employee_forecast.py

📊 Example KPI Columns

    project_delivery_score

    attendance_rate

    quality_score

    teamwork_score

    learning_index

    performance_score (target variable)

📌 Notes

    Random Forest handles tabular KPI regression well for cross-sectional data.

    LSTM is used for sequential forecasting using historical KPI patterns (e.g., monthly updates).

    The synthetic dataset can be replaced with actual HR data for deployment.

👨‍💻 Author: Okes Imoni
Email: jennyimoni@gmail.com
