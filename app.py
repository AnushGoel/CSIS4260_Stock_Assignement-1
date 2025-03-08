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
        .title {text-align: center; font-size: 32px; font-weight: bold; color: #4A90E2;}
        .metric-container {display: flex; justify-content: space-around; padding: 10px;}
        .stMetric {font-size: 18px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Stock Price Forecasting (Extra Trees & Random Forest)</p>', unsafe_allow_html=True)

# Company Selection
selected_company = st.selectbox("Select a company:", valid_companies)
company_data = data[data['name'] == selected_company]
company_data = company_data.sort_index()

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
accuracy = 1 - (mae / np.mean(y_test))  # Estimating accuracy from MAE

# Display Metrics in One Line
st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
st.metric(label="MAE", value=f"{mae:.2f}")
st.metric(label="RMSE", value=f"{rmse:.2f}")
st.metric(label="R² Score", value=f"{r2:.2f}")
st.metric(label="Estimated Accuracy", value=f"{accuracy:.2%}")
st.markdown("</div>", unsafe_allow_html=True)

# Forecast Next 15 Days
future_X = X_test.iloc[-15:, :]
future_forecast = model.predict(future_X)
future_forecast = np.round(future_forecast, 2)  # Round off values

# Display Forecasted Values
forecast_df = pd.DataFrame({"Day": range(1, 16), "Forecasted Close": future_forecast})
st.subheader("15-Day Forecast")
st.dataframe(forecast_df.style.format({"Forecasted Close": "{:.2f}"}))

# Volatility Index Plot
st.subheader("Stock Price Volatility")
company_data['Returns'] = company_data['close'].pct_change()
st.line_chart(company_data['Returns'].dropna())

# Histogram of Returns
st.subheader("Histogram of Daily Returns")
st.plotly_chart(px.histogram(company_data, x='Returns', nbins=50, title='Distribution of Daily Returns'))

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
correlation_matrix = company_data[features + [target]].corr()
st.plotly_chart(px.imshow(correlation_matrix, text_auto=True, title='Correlation Between Features'))

# Bollinger Bands
st.subheader("Bollinger Bands")
company_data['SMA_20'] = company_data['close'].rolling(window=20).mean()
company_data['Upper_Band'] = company_data['SMA_20'] + (company_data['close'].rolling(window=20).std() * 2)
company_data['Lower_Band'] = company_data['SMA_20'] - (company_data['close'].rolling(window=20).std() * 2)

bollinger_fig = go.Figure()
bollinger_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['Upper_Band'], mode='lines', name='Upper Band', line=dict(color='red')))
bollinger_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['SMA_20'], mode='lines', name='20-day SMA', line=dict(color='blue')))
bollinger_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['Lower_Band'], mode='lines', name='Lower Band', line=dict(color='green')))
bollinger_fig.add_trace(go.Scatter(x=company_data.index, y=company_data['close'], mode='lines', name='Close Price', line=dict(color='black')))
st.plotly_chart(bollinger_fig)
