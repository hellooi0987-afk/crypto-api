from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_input(df):

    features = df[
        [
            'close',
            'volume',
            'RSI',
            'EMA',
            'MACD',
            'MACD_Signal',
            'BB_Upper',
            'BB_Lower',
            'Supertrend'
        ]
    ]

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(features)

    last_60 = scaled_data[-60:]

    X = np.array([last_60])

    current_price = df['close'].iloc[-1]

    return X, scaler, current_price
