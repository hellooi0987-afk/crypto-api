import pandas as pd

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

def calculate_macd(data):

    data['MACD'] = (
        data['close'].ewm(span=12, adjust=False).mean()
        -
        data['close'].ewm(span=26, adjust=False).mean()
    )

    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data

def calculate_bollinger(data, window=20):

    data['BB_Middle'] = data['close'].rolling(window=window).mean()

    data['BB_Upper'] = data['BB_Middle'] + (
        2 * data['close'].rolling(window=window).std()
    )

    data['BB_Lower'] = data['BB_Middle'] - (
        2 * data['close'].rolling(window=window).std()
    )

    return data

def calculate_supertrend(data, atr_period=10, multiplier=3):

    tr = []

    for i in range(len(data)):

        hl = data['high'][i] - data['low'][i]

        hc = abs(data['high'][i] - data['close'][i])

        lc = abs(data['low'][i] - data['close'][i])

        tr.append(max(hl, hc, lc))

    data['TR'] = tr

    data['ATR'] = data['TR'].rolling(window=atr_period).mean()

    data['Supertrend'] = data['close'] - multiplier * data['ATR']

    return data

def add_indicators(df):

    df = calculate_rsi(df)

    df = calculate_ema(df)

    df = calculate_macd(df)

    df = calculate_bollinger(df)

    df = calculate_supertrend(df)

    df = df.dropna()

    return df
