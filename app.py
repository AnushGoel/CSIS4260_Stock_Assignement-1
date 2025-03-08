import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_parquet("data.parquet")
    return data

data = load_data()

# Filter companies with at least 180 data points
company_counts = data['name'].value_counts()
valid_companies = company_counts[company_counts >= 180].index.tolist()

data = data[data['name'].isin(valid_companies)]

# Streamlit App Title
st.title("Stock Price Forecasting (Extra Trees & Random Forest)")

# Company Selection
selected_company = st.selectbox("Select a company:", valid_companies)
company_data = data[data['name'] == selected_company]

# Feature Selection
features = ["open", "high", "low", "volume", "EMA_10", "MACD", "ATR_14", "Williams_%R"]
target = "close"

# Prepare Data
X = company_data[features]
y = company_data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Selection
model_choice = st.radio("Select Model:", ["Extra Trees", "Random Forest"], horizontal=True)

if model_choice == "Extra Trees":
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predictions & Metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Metrics
st.subheader("Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"R² Score: {r2:.4f}")

# Forecast Next 15 Days
future_X = X_test.iloc[-15:, :]
future_forecast = model.predict(future_X)

# Display Forecasted Values
forecast_df = pd.DataFrame({"Day": range(1, 16), "Forecasted Close": future_forecast})
st.subheader("15-Day Forecast")
st.dataframe(forecast_df)

# Candlestick Chart
st.subheader("Candlestick Chart")
candlestick_fig = go.Figure(data=[go.Candlestick(x=company_data.index,
                                                  open=company_data['open'],
                                                  high=company_data['high'],
                                                  low=company_data['low'],
                                                  close=company_data['close'])])
st.plotly_chart(candlestick_fig)

# Forecast Plot
st.subheader("Forecasted Prices")
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=range(len(y_test)), y=y_test, mode='lines', name='Actual Close Price'))
forecast_fig.add_trace(go.Scatter(x=range(len(y_test)), y=y_pred, mode='lines', name='Predicted Close Price'))
forecast_fig.add_trace(go.Scatter(x=range(len(y_test), len(y_test) + 15), y=future_forecast, mode='lines', name='Forecasted Close Price', line=dict(dash='dot')))
st.plotly_chart(forecast_fig)

# Seasonal Plot
st.subheader("Seasonal Trends")
decompose_result = seasonal_decompose(company_data[target], period=30, model='additive', extrapolate_trend='freq')
seasonal_fig = go.Figure()
seasonal_fig.add_trace(go.Scatter(x=company_data.index, y=decompose_result.trend, mode='lines', name='Trend'))
seasonal_fig.add_trace(go.Scatter(x=company_data.index, y=decompose_result.seasonal, mode='lines', name='Seasonality'))
st.plotly_chart(seasonal_fig)

# Financial Metrics Chart
st.subheader("Financial Metrics")
metrics_fig = go.Figure()
metrics_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['EMA_10'], mode='lines', name='EMA 10'))
metrics_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['MACD'], mode='lines', name='MACD'))
metrics_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['ATR_14'], mode='lines', name='ATR 14'))
st.plotly_chart(metrics_fig)

# Moving Average Chart
st.subheader("Moving Averages")
company_data['SMA_50'] = company_data['close'].rolling(window=50).mean()
company_data['SMA_200'] = company_data['close'].rolling(window=200).mean()
moving_avg_fig = go.Figure()
moving_avg_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['close'], mode='lines', name='Close Price'))
moving_avg_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['SMA_50'], mode='lines', name='50-day SMA'))
moving_avg_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['SMA_200'], mode='lines', name='200-day SMA'))
st.plotly_chart(moving_avg_fig)
