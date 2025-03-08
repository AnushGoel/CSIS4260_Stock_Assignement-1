import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import joblib
from statsmodels.tsa.seasonal import seasonal_decompose

# Set Page Configurations
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_parquet("data.parquet")
    data = data.dropna()  # Drop null values
    return data

data = load_data()

# Filter companies with at least 180 data points
company_counts = data['name'].value_counts()
valid_companies = company_counts[company_counts >= 180].index.tolist()

data = data[data['name'].isin(valid_companies)]

# Streamlit App Title with Styling
st.markdown("""
    <style>
        .title {text-align: center; font-size: 36px; font-weight: bold; color: #4A90E2;}
        .metric-container {display: flex; justify-content: space-around; padding: 10px;}
        .stMetric {font-size: 18px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">📊 Advanced Stock Market Forecasting Dashboard</p>', unsafe_allow_html=True)

# Layout Setup
col1, col2 = st.columns([1, 3])

with col1:
    selected_company = st.selectbox("Select a Company:", valid_companies)
    model_choice = st.radio("Select Forecasting Model:", ["Extra Trees", "Random Forest"], horizontal=True)
    
with col2:
    st.write("### Stock Data Overview")
    company_data = data[data['name'] == selected_company]
    company_data = company_data.sort_index()
    company_data.index = pd.to_datetime(company_data.index)  # Ensure index is in datetime format
    st.dataframe(company_data.tail(10).style.format(precision=2))

# Feature Selection
features = ["open", "high", "low", "volume", "EMA_10", "MACD", "ATR_14", "Williams_%R"]
target = "close"

# Prepare Data
X = company_data[features]
y = company_data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Selection
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
accuracy = 1 - (mae / np.mean(y_test))

# Display Metrics in One Row
st.markdown("---")
st.subheader("📈 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")
col4.metric("Accuracy", f"{accuracy:.2%}")

# Forecast Next 15 Days
forecast_dates = company_data.index[-15:]  # Use actual dataset dates
future_X = X.iloc[-15:, :]
future_forecast = model.predict(future_X)
future_forecast = np.round(future_forecast, 2)

# Display Forecasted Values with Dates
forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted Close": future_forecast})
st.subheader("📅 15-Day Forecast")
st.dataframe(forecast_df.style.format({"Forecasted Close": "{:.2f}"}))

# Forecasted Chart
st.subheader("📊 Forecasted Stock Price")
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecasted Close"], mode='lines+markers', name='Forecasted Close Price', line=dict(color='red')))
st.plotly_chart(forecast_fig, use_container_width=True)

# Candlestick Chart
st.subheader("📊 Candlestick Chart")
candlestick_fig = go.Figure(data=[go.Candlestick(x=company_data.index,
                                                  open=company_data['open'],
                                                  high=company_data['high'],
                                                  low=company_data['low'],
                                                  close=company_data['close'])])
st.plotly_chart(candlestick_fig, use_container_width=True)

# Volume Analysis
st.subheader("📊 Trading Volume Over Time")
st.line_chart(company_data['volume'])

# RSI Indicator
st.subheader("📈 Relative Strength Index (RSI)")
delta = company_data['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
st.line_chart(rsi)

# MACD Indicator
st.subheader("📉 MACD Indicator")
company_data['MACD_Line'] = company_data['close'].ewm(span=12, adjust=False).mean() - company_data['close'].ewm(span=26, adjust=False).mean()
company_data['Signal_Line'] = company_data['MACD_Line'].ewm(span=9, adjust=False).mean()
st.line_chart(company_data[['MACD_Line', 'Signal_Line']])
