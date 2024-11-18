import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import yfinance as yf

app = Flask(__name__)

DATA_DIR = os.path.expanduser("~/Documents/archive-2/stocks")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 60
PREDICT_DAYS = 30
EPOCHS = 15
MC_SAMPLES = 10

CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    "LTC-USD", "LINK-USD", "UNI-USD", "ATOM-USD", "TRX-USD"
]

CRYPTO_NAMES = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "BNB",
    "SOL-USD": "Solana", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche", "DOT-USD": "Polkadot",
    "MATIC-USD": "Polygon", "LTC-USD": "Litecoin", "LINK-USD": "Chainlink",
    "UNI-USD": "Uniswap", "ATOM-USD": "Cosmos", "TRX-USD": "TRON"
}

training_progress = {}


def is_crypto(ticker):
    return ticker.endswith("-USD") or ticker in CRYPTO_TICKERS


def safe_filename(ticker):
    """Convert ticker to safe filename (BTC-USD → BTC_USD)."""
    return ticker.replace("-", "_")


class ProgressCallback(Callback):
    def __init__(self, ticker, total_epochs):
        self.ticker = ticker
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        training_progress[self.ticker] = {
            "epoch": epoch + 1,
            "total": self.total_epochs,
            "status": "training"
        }


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def load_stock_data(ticker):
    # Try yfinance first for live data
    try:
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if len(df) > 100:
            df = df.reset_index()
            # Flatten multi-level columns if present
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            return df, "live"
    except Exception:
        pass

    # Fall back to local CSV
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df, "csv"

    return None, None


def prepare_features(df):
    df = df.copy()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["Vol_norm"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-10)
    df = df.dropna().reset_index(drop=True)
    return df


def build_model(seq_len, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def mc_predict(model, sequence, n_samples=30):
    """Monte Carlo Dropout: run n_samples forward passes with dropout active."""
    preds = np.array([model(sequence, training=True).numpy() for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0)


def train_and_predict(ticker):
    df, source = load_stock_data(ticker)
    if df is None or len(df) < SEQUENCE_LENGTH + 60:
        return None

    df = prepare_features(df)

    feature_cols = ["Close", "MA_20", "MA_50", "RSI", "Vol_norm"]
    n_features = len(feature_cols)

    # Scale each feature independently
    scalers = {}
    scaled_data = np.zeros((len(df), n_features))
    for i, col in enumerate(feature_cols):
        scaler = MinMaxScaler()
        scaled_data[:, i] = scaler.fit_transform(df[[col]]).flatten()
        scalers[col] = scaler

    close_scaler = scalers["Close"]

    # Build sequences
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    fname = safe_filename(ticker)
    model_path = os.path.join(MODELS_DIR, f"{fname}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{fname}_scalers.pkl")
    from_cache = False

    training_progress[ticker] = {"epoch": 0, "total": EPOCHS, "status": "training"}

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        close_scaler = scalers["Close"]
        training_progress[ticker] = {"epoch": EPOCHS, "total": EPOCHS, "status": "cached"}
        from_cache = True
    else:
        model = build_model(SEQUENCE_LENGTH, n_features)
        cb = ProgressCallback(ticker, EPOCHS)
        model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=32,
            validation_split=0.1,
            callbacks=[cb], verbose=0
        )
        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scalers, f)

    # Test set predictions
    test_preds_scaled = model.predict(X_test, verbose=0)
    test_preds = close_scaler.inverse_transform(test_preds_scaled)

    # Forecast with MC Dropout for uncertainty bands
    current_seq = scaled_data[-SEQUENCE_LENGTH:].copy()
    future_means, future_lowers, future_uppers = [], [], []

    for _ in range(PREDICT_DAYS):
        seq_input = current_seq.reshape(1, SEQUENCE_LENGTH, n_features)
        mean_pred, std_pred = mc_predict(model, seq_input, MC_SAMPLES)
        val = float(mean_pred[0, 0])
        std = float(std_pred[0, 0])
        future_means.append(val)
        future_lowers.append(val - 1.5 * std)
        future_uppers.append(val + 1.5 * std)
        next_row = current_seq[-1].copy()
        next_row[0] = val
        current_seq = np.vstack([current_seq[1:], next_row])

    future_prices = close_scaler.inverse_transform(np.array(future_means).reshape(-1, 1)).flatten()
    lower_prices = close_scaler.inverse_transform(np.array(future_lowers).reshape(-1, 1)).flatten()
    upper_prices = close_scaler.inverse_transform(np.array(future_uppers).reshape(-1, 1)).flatten()

    last_date = df["Date"].iloc[-1]
    if is_crypto(ticker):
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICT_DAYS)
    else:
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PREDICT_DAYS)

    # Return last 365 days of history
    n_hist = min(365, len(df))
    hist_dates = df["Date"].iloc[-n_hist:].dt.strftime("%Y-%m-%d").tolist()
    hist_prices = df["Close"].iloc[-n_hist:].round(2).tolist()
    hist_volume = [int(v) for v in df["Volume"].iloc[-n_hist:].tolist()]

    test_start_idx = len(df) - len(y_test)
    test_dates = df["Date"].iloc[test_start_idx:].dt.strftime("%Y-%m-%d").tolist()
    test_prices = test_preds.flatten().round(2).tolist()

    future_dates_list = [d.strftime("%Y-%m-%d") for d in future_dates]
    future_prices_list = future_prices.round(2).tolist()
    lower_prices_list = lower_prices.round(2).tolist()
    upper_prices_list = upper_prices.round(2).tolist()

    current_price = float(df["Close"].iloc[-1])
    predicted_price = float(future_prices[-1])
    change = round(((predicted_price - current_price) / current_price) * 100, 2)

    training_progress[ticker] = {"epoch": EPOCHS, "total": EPOCHS, "status": "done"}

    return {
        "ticker": ticker,
        "asset_name": CRYPTO_NAMES.get(ticker, ticker),
        "is_crypto": is_crypto(ticker),
        "source": source,
        "from_cache": from_cache,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "change_pct": change,
        "direction": "UP" if change > 0 else "DOWN",
        "hist_dates": hist_dates,
        "hist_prices": hist_prices,
        "hist_volume": hist_volume,
        "test_dates": test_dates,
        "test_prices": test_prices,
        "future_dates": future_dates_list,
        "future_prices": future_prices_list,
        "lower_prices": lower_prices_list,
        "upper_prices": upper_prices_list,
        "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "total_rows": len(df)
    }


def get_available_tickers():
    files = [f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    return sorted(files)


@app.route("/")
def index():
    tickers = get_available_tickers()
    popular = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "BABA"]
    return render_template("index.html", tickers=tickers, popular=popular,
                           crypto_tickers=CRYPTO_TICKERS, crypto_names=CRYPTO_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    result = train_and_predict(ticker)
    if result is None:
        return jsonify({"error": f"Could not find or process data for {ticker}"}), 404
    return jsonify(result)


@app.route("/progress/<ticker>")
def progress(ticker):
    ticker = ticker.upper()
    info = training_progress.get(ticker, {"epoch": 0, "total": EPOCHS, "status": "waiting"})
    return jsonify(info)


@app.route("/tickers")
def tickers():
    return jsonify(get_available_tickers())


if __name__ == "__main__":
    app.run(debug=True, port=5007)
