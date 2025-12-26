# Hybrid ML–Econometric Framework for Bitcoin Implied Volatility Change Forecasting

**Author:** Anuj Pal  
**Context:** Doctoral Research Proposal (Quantitative Finance / Cryptoeconomics)

---

## 📌 Project Overview
This repository contains the full codebase for a hybrid machine learning–econometric framework designed to forecast **one-day changes in Bitcoin implied volatility (DVOL)**.

The framework integrates:
- **On-chain metrics:** NUPL (Net Unrealized Profit/Loss), SOPR (Spent Output Profit Ratio)
- **Signal processing:** Continuous Wavelet Transforms (CWT) for multi-scale regime detection
- **Macro-financial controls:** CPI, policy rate momentum, and DXY
- **Nonlinear modeling:** XGBoost benchmarked against an AR(1)–GARCH(1,1) baseline

The primary objective is to study the distinction between **statistical predictability** and **economic tradability** in crypto-derivative markets.

---

## 📊 Summary of Empirical Results
Key results reproduced by this codebase:

| Metric | Result | Interpretation |
|------|--------|----------------|
| **Out-of-Sample R² (XGBoost)** | **0.0527** | Positive predictive power in short-horizon implied volatility *change* forecasting |
| **Out-of-Sample R² (GARCH)**   | **-0.0107** | Classical volatility models fail to capture belief-driven implied volatility *change* dynamics |
| **Strategy Sharpe (Spot-Normalized)** | **-13.40** | Statistical predictability does not translate into naive economic profitability |

> **Interpretation:**  
> The results highlight a clear separation between statistical predictability and economic tradability. 
---

## 📂 Repository Structure
- `exploratory_results.ipynb` – Exploratory research notebook used to generate empirical results, figures, and diagnostics  
- `main.py` – Clean, script-based implementation of the same methodology for reproducibility and inspection  
- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation  

### Reproducibility Note
The `.py` script is a direct transcription of the logic implemented in the notebook.  
Only cosmetic changes (comment cleanup and removal of informal console text) were made for academic clarity.  
**Model specification, data processing, and reported results are unchanged.**

---

## 📁 Data Availability
The dataset used in this study combines:
- On-chain metrics
- Market prices and implied volatility indices
- Macroeconomic indicators
- Sentiment indices

Due to **licensing, terms-of-service, and usage restrictions** associated with some data sources, the raw dataset is **not redistributed** in this repository.

Researchers interested in replication can reconstruct equivalent datasets from the original providers or public sources by following the feature engineering and modeling pipeline implemented in the code.
Execution requires access to a locally reconstructed dataset matching the column schema used in the code.

## 📑 Dataset Schema

The empirical analysis uses a daily time-series dataset constructed from market, on-chain,
sentiment, and macroeconomic sources. The core columns expected by the pipeline are:

| Column | Description | Frequency |
|------|------------|-----------|
| `date` | Observation date | Daily |
| `open` | Bitcoin opening price (USD) | Daily |
| `high` | Bitcoin high price (USD) | Daily |
| `low` | Bitcoin low price (USD) | Daily |
| `close` | Bitcoin closing price (USD) | Daily |
| `volume` | Bitcoin spot trading volume | Daily |
| `iv` | Bitcoin implied volatility index (DVOL) | Daily |
| `nupl` | Net Unrealized Profit/Loss (on-chain) | Daily |
| `sopr` | Spent Output Profit Ratio (on-chain) | Daily |
| `miner_btc_outflow` | BTC outflow from miners | Daily |
| `miner_tx_count` | Number of miner transactions | Daily |
| `num_tweets` | Count of crypto-related social media posts | Daily |
| `avg_sentiment` | Average sentiment score | Daily |
| `sentiment_variance` | Cross-sectional sentiment dispersion | Daily |
| `bitcoin_trend` | Google Trends index for Bitcoin | Daily |
| `DXY` | US Dollar Index | Daily |
| `CPI` | Consumer Price Index | Monthly (forward-filled) |
| `Fed_Rate` | Policy interest rate | Monthly (forward-filled) |

All additional features (e.g., realized volatility measures, wavelet coefficients,
technical indicators, and interaction terms) are generated internally by the modeling
pipeline and documented directly in `main.py`.

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python main.py
