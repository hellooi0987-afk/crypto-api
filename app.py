from fastapi import FastAPI
from tf_keras.models import load_model
import numpy as np
import pandas as pd

from utils.data_fetcher import fetch_latest_data
from utils.indicators import add_indicators
from utils.preprocess import preprocess_input

app = FastAPI()

# Load trained model
model = load_model("models/btcusdt_model.h5")

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Crypto AI API Live"
    }

@app.get("/predict/{symbol}")
def predict(symbol: str):

    try:

        # Fetch latest data
        df = fetch_latest_data(symbol)

        # Add indicators
        df = add_indicators(df)

        # Preprocess
        X, scaler, current_price = preprocess_input(df)

        # Predict
        prediction = model.predict(X)

        predicted_price = scaler.inverse_transform(
            np.concatenate(
                [prediction.reshape(-1, 1), np.zeros((len(prediction), 8))],
                axis=1
            )
        )[:, 0][0]

        signal = "BUY" if predicted_price > current_price else "SELL"

        return {
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "predicted_price": round(float(predicted_price), 2),
            "signal": signal
        }

    except Exception as e:
        return {
            "error": str(e)
        }
