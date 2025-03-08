# 📊 CSIS 4260 – Assignment 1: Stock Price Prediction & Analysis

## 📝 Overview
This project focuses on **storing, analyzing, and forecasting stock prices** for S&P 500 companies using time-series data. The dataset spans from **2013-02-08 to 2018-02-07** and consists of **619,040 rows**.

The project is divided into **three main parts:**
1. **Storage Format Benchmarking** – Compare CSV vs Parquet.
2. **Data Analysis & Prediction** – Implement technical indicators and predictive models.
3. **Visual Dashboard** – Create an interactive dashboard for insights.

---

## 📌 Project Structure
```
📂 CSIS4260-Stock-Prediction
│── 📂 data/                     # Raw and processed data files
│── 📂 notebooks/                 # Jupyter notebooks for analysis
│── 📂 src/                       # Source code for storage, analysis, and prediction
│── 📂 dashboard/                 # Streamlit/Dash dashboard implementation
│── main.py                      # Main script for the Streamlit dashboard
│── requirements.txt             # Python dependencies
│── README.md                    # Project documentation (this file)
```

---

## 📂 Dataset Details
- **Source:** Provided by course instructor
- **Companies:** 505 S&P 500 companies
- **Columns:** `date`, `open`, `high`, `low`, `close`, `volume`, `name`, `EMA_10`, `MACD`, `Williams_%R`, `ATR_14`
- **Size:** ~29MB CSV (619,040 rows)

---

## 🏗️ Part 1: Storage & Retrieval ([🔗 View Notebook](https://github.com/AshmithaJagadish/CSIS-4260-Stock-Project/blob/main/Part-A%20.ipynb))
**Goal:** Compare CSV vs Parquet storage formats for efficiency.
- ✅ Benchmark **read/write speeds**
- ✅ Compare **file size reduction** using compression
- ✅ Test performance at **1x, 10x, and 100x** dataset size

**Tools Used:** Pandas, PyArrow, Fastparquet

---

## 📊 Part 2: Data Manipulation & Prediction ([🔗 View Notebook](https://github.com/AshmithaJagadish/CSIS-4260-Stock-Project/blob/main/Part-B.ipynb))
**Goal:** Enhance dataset with technical indicators & predict next-day closing prices.
- ✅ Implement **4+ technical indicators** (EMA, MACD, ATR, Williams %R)
- ✅ Compare **Pandas vs Polars** for performance
- ✅ Use **Extra Trees Regressor & XGBoost** for stock price forecasting
- ✅ Train models with an **80-20 train-test split**

**Libraries Used:** Pandas, Polars, Scikit-learn, XGBoost

---

## 🎨 Part 3: Interactive Dashboard ([🔗 View Live Dashboard](https://mainpy-mvymenbdcxfaks6c573vvp.streamlit.app/))
**Goal:** Develop a dashboard to visualize benchmarking and predictions.
- ✅ **Section A:** Display storage & performance benchmarks
- ✅ **Section B:** Enable **company selection** to visualize predictions
- ✅ **Candlestick charts** for historical stock data

**Tools Used:** Streamlit, Plotly, Dash

---

## 🚀 Setup & Installation
### **🔹 Step 1: Clone Repository**
```bash
git clone https://github.com/your-username/CSIS4260-Stock-Prediction.git
cd CSIS4260-Stock-Prediction
```

### **🔹 Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **🔹 Step 3: Run the Streamlit Dashboard**
```bash
streamlit run main.py
```

---

## 📌 Future Enhancements
- [ ] Optimize model for better accuracy
- [ ] Experiment with LSTM for deep learning forecasting
- [ ] Deploy dashboard on **Streamlit Cloud**

---

## 👨‍💻 Contributors
- **[Your Name]** – Data Processing & Model Implementation
- **[Your Name]** – Dashboard Development

📩 For questions, contact: `your.email@example.com`

---

## 📜 License
This project is for educational purposes under **CSIS 4260** and is not for commercial use.

---

🌟 **If you find this project helpful, give it a star on GitHub!** ⭐
