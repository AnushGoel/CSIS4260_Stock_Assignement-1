import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import joblib

# Set Page Configurations
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# Define tickers for five companies
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

@st.cache_data
def load_data():
    all_data = []
    # Download data for each ticker from Yahoo Finance
    for ticker in TICKERS:
        df = yf.download(ticker, start="2016-01-01")
        if df.empty:
            continue
        # Reset index to get date as a column
        df.reset_index(inplace=True)
        # Ensure date column is in date format (dropping time for consistency)
        df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        df = df[df['date'] > pd.to_datetime('1970-01-01').date()]
        
        # Convert column names to lower-case for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # --- Compute Technical Indicators ---
        # EMA_10 using the close price
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        
        # MACD: difference between EMA12 and EMA26
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        
        # ATR_14: Average True Range over 14 days
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = (df['high'] - df['close'].shift(1)).abs()
        df['l-pc'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['ATR_14'] = df['tr'].rolling(window=14).mean()
        
        # Williams %R indicator (period = 14)
        period = 14
        df['highest_high'] = df['high'].rolling(window=period).max()
        df['lowest_low'] = df['low'].rolling(window=period).min()
        df['Williams_%R'] = -100 * ((df['highest_high'] - df['close']) / (df['highest_high'] - df['lowest_low']))
        
        # Drop temporary columns
        df.drop(columns=['h-l', 'h-pc', 'l-pc', 'tr', 'highest_high', 'lowest_low'], inplace=True, errors='ignore')
        
        # Add company ticker as a column
        df['name'] = ticker
        
        all_data.append(df)
        
    # Combine data for all companies and drop any remaining missing values
    if all_data:
        data = pd.concat(all_data, ignore_index=True)
        data.dropna(inplace=True)
        return data
    else:
        return pd.DataFrame()

# Load data from Yahoo Finance
data = load_data()

# Filter companies with at least 180 data points
valid_companies = data['name'].value_counts()[lambda x: x >= 180].index.tolist()
data = data[data['name'].isin(valid_companies)]

# Streamlit App Title
st.title("📊 Advanced Stock Market Forecasting Dashboard")

# Layout Setup
col1, col2 = st.columns([1, 3])
with col1:
    selected_company = st.selectbox("Select a Company:", valid_companies)
    model_choice = st.radio("Select Forecasting Model:", ["Extra Trees", "Random Forest"], horizontal=True)
with col2:
    st.subheader("Stock Data Overview")
    company_data = data[data['name'] == selected_company].sort_values(by='date')
    st.dataframe(company_data.tail(10).style.format(precision=2))

# Feature Selection: using open, high, low, volume, and technical indicators
features = ["open", "high", "low", "volume", "EMA_10", "MACD", "ATR_14", "Williams_%R"]
target = "close"

# Prepare Data for modeling
X = company_data[features]
y = company_data[target]

# Train-Test Split (without shuffling to preserve time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Selection
if model_choice == "Extra Trees":
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions & Metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
accuracy = explained_variance_score(y_test, y_pred)

# Display Metrics in one styled row
st.subheader("📈 Model Performance Metrics")
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-around; font-size: 18px; font-weight: bold; padding: 12px; background-color: #f8f9fa; border-radius: 8px;">
        <div>📉 MAE: <span style='color: red;'>{mae:.2f}</span></div>
        <div>📊 RMSE: <span style='color: blue;'>{rmse:.2f}</span></div>
        <div>📈 R² Score: <span style='color: green;'>{r2:.2f}</span></div>
        <div>✅ Accuracy: <span style='color: purple;'>{accuracy:.2%}</span></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Forecast Next 15 Days: using the most recent 15 data points as features to predict the "close" value
forecast_dates = company_data['date'].iloc[-15:].values
future_X = X.iloc[-15:, :]
future_forecast = np.round(model.predict(future_X), 2)
forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted Close": future_forecast})

# Display Forecasted Values with a styled table
st.subheader("📅 15-Day Forecast")
st.markdown("""
<style>
    .styled-table {
        width: 50%;
        margin-left: auto;
        margin-right: auto;
        border-collapse: collapse;
        border: 1px solid #ddd;
    }
    .styled-table th, .styled-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .styled-table th {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<table class='styled-table'>
<tr><th>Date</th><th>Forecasted Close</th></tr>
{''.join(f'<tr><td>{date}</td><td>{value:.2f}</td></tr>' for date, value in zip(forecast_df['Date'], forecast_df['Forecasted Close']))}
</table>
""", unsafe_allow_html=True)

# Forecasted Chart
st.subheader("📊 Forecasted Stock Price")
st.plotly_chart(px.line(forecast_df, x="Date", y="Forecasted Close", title="Forecasted Close Price"))

# Candlestick Chart
st.subheader("📊 Candlestick Chart")
st.plotly_chart(go.Figure(data=[go.Candlestick(x=company_data['date'],
                                               open=company_data['open'],
                                               high=company_data['high'],
                                               low=company_data['low'],
                                               close=company_data['close'])]))

# Volume Analysis
st.subheader("📊 Trading Volume Over Time")
st.line_chart(company_data.set_index('date')['volume'])

# Williams %R Indicator
st.subheader("📉 Williams %R Indicator")
st.line_chart(company_data.set_index('date')['Williams_%R'])

# Additional Insights - ATR & MACD Analysis
st.subheader("📊 ATR & MACD Analysis")
st.line_chart(company_data.set_index('date')[['ATR_14', 'MACD']])
