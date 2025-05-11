
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Simulate 24 months of KPI data for one employee
np.random.seed(42)
time_steps = 24

time_data = {
    'project_delivery_score': np.random.normal(75, 5, time_steps),
    'attendance_rate': np.random.normal(90, 3, time_steps),
    'quality_score': np.random.normal(80, 4, time_steps),
    'teamwork_score': np.random.normal(78, 3, time_steps),
    'learning_index': np.random.normal(70, 5, time_steps)
}

df_time = pd.DataFrame(time_data)
df_time['performance_score'] = (
    0.3 * df_time['project_delivery_score'] +
    0.2 * df_time['attendance_rate'] +
    0.2 * df_time['quality_score'] +
    0.15 * df_time['teamwork_score'] +
    0.15 * df_time['learning_index'] +
    np.random.normal(0, 2, time_steps)
)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_time)

# Prepare LSTM input/output
X, y = [], []
n_input = 3
for i in range(n_input, len(scaled_data)):
    X.append(scaled_data[i-n_input:i, :-1])
    y.append(scaled_data[i, -1])
X, y = np.array(X), np.array(y)

# Define and train model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_input, 5)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=1)

# Forecast next time step
last_input = scaled_data[-n_input:, :-1]
last_input = np.expand_dims(last_input, axis=0)
forecast_scaled = model.predict(last_input)[0][0]

# Inverse transform
placeholder = np.zeros((1, scaled_data.shape[1]))
placeholder[0, -1] = forecast_scaled
forecast_actual = scaler.inverse_transform(placeholder)[0, -1]

print(f"Forecasted Performance Score for Next Month: {forecast_actual:.2f}")
