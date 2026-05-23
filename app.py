from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

API_KEY = 'bg_7f6c0ca03bff5fc7bc850ce03b26c54c'
API_PASS = 'humaiz9999'
BITGET_URL = 'https://api.bitget.com'
TIME_STEP = 60

def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_ema(data, span=20):
    data['EMA'] = data['close'].ewm(span=span, adjust=False).mean()
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['MACD'] = data['close'].ewm(span=short_window, adjust=False).mean() - \
                   data['close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_bollinger_bands(data, window=20):
    data['BB_Middle'] = data['close'].rolling(window=window).mean()
    data['BB_Upper'] = data['BB_Middle'] + \
                       (2 * data['close'].rolling(window=window).std())
    data['BB_Lower'] = data['BB_Middle'] - \
                       (2 * data['close'].rolling(window=window).std())
    return data

def calculate_supertrend(data, atr_period=10, multiplier=3):
    data['TR'] = data[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'],
                      abs(x['high'] - x['close']),
                      abs(x['low'] - x['close'])), axis=1)
    data['ATR'] = data['TR'].rolling(window=atr_period).mean()
    data['Supertrend'] = (data['close'] - multiplier * data['ATR']).where(
        data['close'] < (data['close'] - multiplier * data['ATR']),
        data['close'] + multiplier * data['ATR'])
    return data

def fetch_live_data(symbol):
    end_time = str(int(time.time() * 1000))
    start_time = str(int((time.time() - (90 * 24 * 60 * 60)) * 1000))
    url = f"{BITGET_URL}/api/v2/spot/market/history-candles?symbol={symbol}&granularity=4h&startTime={start_time}&endTime={end_time}&limit=200"
    headers = {
        'Content-Type': 'application/json',
        'ACCESS-KEY': API_KEY,
        'ACCESS-PASSPHRASE': API_PASS,
        'ACCESS-TIMESTAMP': str(int(time.time() * 1000)),
    }
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'],
                columns=['timestamp','open','high','low','close',
                         'volume','quote_volume','count'])
            df[['high','low','close','open']] = df[['high','low','close','open']]\
                .apply(pd.to_numeric, errors='coerce')
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[['timestamp','open','high','low','close','volume']]
    return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "CryptoNeuron API Running", "version": "1.0"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid JSON request"}), 400

        symbol = data.get('symbol', 'BTCUSDT').upper()
        leverage = int(data.get('leverage', 10))
        margin = int(data.get('margin', 100))

        model_path = f"models/{symbol.lower()}_model.h5"
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model not found for {symbol}. Available: BTCUSDT, ETHUSDT"}), 404

        df = fetch_live_data(symbol)
        if df is None:
            return jsonify({"error": "Failed to fetch market data from Bitget"}), 500

        df = calculate_rsi(df)
        df = calculate_ema(df)
        df = calculate_macd(df)
        df = calculate_bollinger_bands(df)
        df = calculate_supertrend(df)
        df = df.dropna()

        if len(df) < TIME_STEP + 2:
            return jsonify({"error": "Not enough data for prediction"}), 400

        scaler = MinMaxScaler()
        features = ['close','volume','RSI','EMA','MACD',
                    'MACD_Signal','BB_Upper','BB_Lower','Supertrend']
        scaled = scaler.fit_transform(df[features])

        X_input = scaled[-TIME_STEP:].reshape(1, TIME_STEP, len(features))

        model = load_model(model_path, compile=False)
        pred_scaled = model.predict(X_input)

        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled[0][0]
        predicted_price = scaler.inverse_transform(dummy)[0][0]

        current_price = float(df['close'].iloc[-1])
        last_row = df.iloc[-1]

        if predicted_price > current_price:
            action = "BUY"
            target = predicted_price * 1.02
            stop_loss = current_price * 0.98
            liquidation = current_price - (current_price / leverage)
            reason = "LSTM predicted upward price movement based on RSI, MACD & Bollinger Bands"
        else:
            action = "SELL"
            target = predicted_price * 0.98
            stop_loss = current_price * 1.02
            liquidation = current_price + (current_price / leverage)
            reason = "LSTM predicted downward price movement based on RSI, MACD & Bollinger Bands"

        response = jsonify({
            "symbol": symbol,
            "current_price": round(current_price, 4),
            "predicted_price": round(float(predicted_price), 4),
            "signal": {
                "action": action,
                "entry_price": round(current_price, 4),
                "target_price": round(float(target), 4),
                "stop_loss": round(float(stop_loss), 4),
                "liquidation_price": round(float(liquidation), 4),
                "reason": reason,
                "margin": margin
            },
            "indicators": {
                "rsi": round(float(last_row['RSI']), 2),
                "ema": round(float(last_row['EMA']), 4),
                "macd": round(float(last_row['MACD']), 4),
                "bb_upper": round(float(last_row['BB_Upper']), 4),
                "bb_lower": round(float(last_row['BB_Lower']), 4),
            }
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
