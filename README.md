# 📈 Stock Price & Crypto Predictor

A deep learning web app that predicts future stock and cryptocurrency prices using an LSTM neural network. Select any of 5,884 stock tickers or 15 cryptocurrencies — the model fetches live data, trains in real time, and forecasts the next 30 days with uncertainty bands.

## Demo

Select a stock or crypto → AI fetches live data → LSTM trains with epoch progress bar → See 30-day forecast with confidence bands, volume chart, and buy/sell signal.

## Results

| Metric | Detail |
|--------|--------|
| Stock Dataset | 5,884 tickers (historical daily prices) |
| Crypto | 15 coins via Yahoo Finance (live data) |
| Model | LSTM (2 layers, 64 + 32 units) |
| Input Features | Close, MA20, MA50, RSI, Volume (5 features) |
| Forecast Window | 30 days |
| Input Sequence | 60 trading days |
| Training Split | 80% train / 20% test |

## Features

- **Stocks & Crypto tabs** — switch between 5,884 stocks and 15 cryptocurrencies
- **Live data** via yfinance — real-time prices, not frozen historical data
- **Multi-feature LSTM** — trained on Close price, MA20, MA50, RSI, and Volume
- **Uncertainty bands** — Monte Carlo Dropout shows forecast confidence range
- **Model caching** — first prediction trains (~20s), every run after is instant
- **Epoch progress bar** — watch the model train in real time (Epoch 1/15 → 15/15)
- **Date range selector** — zoom chart to 1M / 3M / 6M / 1Y / All
- **Volume chart** — bar chart below the price chart
- **Buy/Sell signal** — based on predicted 30-day trend
- **Live/CSV badge** — shows whether data came from yfinance or local CSV fallback
- Dark finance-style UI

## Supported Cryptocurrencies

| Coin | Ticker | Coin | Ticker |
|------|--------|------|--------|
| Bitcoin | BTC-USD | Dogecoin | DOGE-USD |
| Ethereum | ETH-USD | Avalanche | AVAX-USD |
| BNB | BNB-USD | Polkadot | DOT-USD |
| Solana | SOL-USD | Polygon | MATIC-USD |
| XRP | XRP-USD | Litecoin | LTC-USD |
| Cardano | ADA-USD | Chainlink | LINK-USD |
| Uniswap | UNI-USD | Cosmos | ATOM-USD |
| TRON | TRX-USD | | |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Model | LSTM (Long Short-Term Memory) |
| Uncertainty | Monte Carlo Dropout |
| Live Data | yfinance (Yahoo Finance API) |
| Data Processing | Pandas, NumPy, Scikit-Learn |
| Web Framework | Flask |
| Frontend | HTML, CSS, Chart.js |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/stock-price-predictor.git
cd stock-price-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Download the stock dataset**

For offline stock data, download from [Kaggle — Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) and place the CSV files in:
```
~/Documents/archive-2/stocks/
```
Crypto works without any dataset — it pulls live from Yahoo Finance automatically.

**4. Start the app**
```bash
python3 app.py
```

**5. Open in browser**
```
http://127.0.0.1:5007
```

## Project Structure

```
stock-price-predictor/
├── app.py               # Flask server, LSTM training, yfinance, MC Dropout
├── models/              # Cached trained models per ticker (auto-generated)
├── templates/
│   └── index.html       # Stocks/Crypto tabs, progress bar, Chart.js charts
├── static/
│   └── style.css        # Dark finance theme
└── requirements.txt
```

## How It Works

1. User selects a stock or crypto ticker
2. **yfinance** fetches live historical data (falls back to local CSV for stocks)
3. Five features are computed: Close, MA20, MA50, RSI-14, normalised Volume
4. Each feature is scaled independently with MinMaxScaler
5. Sequences of 60 days are used as LSTM input windows
6. **LSTM model** (2 layers) learns temporal price patterns over 15 epochs
7. Model is evaluated on the test set (last 20% of data)
8. **Monte Carlo Dropout** runs 10 forward passes per forecast step to generate uncertainty bands
9. Model is cached to disk — instant on second load
10. Chart.js displays historical prices, model fit, 30-day forecast, confidence bands, and volume

## Model Architecture

```
Input: (60 days, 5 features)
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32)
  → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1)
Output: next day Close price (scaled)
```

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Crop Disease Detector | EfficientNetV2 — 15 plant disease categories | [crop-disease-detector](https://github.com/manny2341/crop-disease-detector) |
| Emotion Detection | Real-time CNN webcam emotion recognition | [Emotion-Detection](https://github.com/manny2341/Emotion-Detection) |
| Image Classifier | EfficientNetV2 — 1,000 categories | [image-classifier](https://github.com/manny2341/image-classifier) |
| Wildfire Detection | YOLOv8 on Sentinel-2 satellite imagery | [wildfire-detection](https://github.com/manny2341/wildfire-detection-and-monitoring-) |

## Author

[@manny2341](https://github.com/manny2341)
