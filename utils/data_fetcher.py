import requests
import pandas as pd
import time

BITGET_URL = "https://api.bitget.com"

def fetch_latest_data(symbol="BTCUSDT"):

    end_time = str(int(time.time() * 1000))
    start_time = str(int((time.time() - (90 * 24 * 60 * 60)) * 1000))

    url = f"{BITGET_URL}/api/v2/spot/market/history-candles?symbol={symbol}&granularity=4h&startTime={start_time}&endTime={end_time}&limit=100"

    response = requests.get(url)

    data = response.json()

    df = pd.DataFrame(
        data["data"],
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "count"
        ]
    )

    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df
