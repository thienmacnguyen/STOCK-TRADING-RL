import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t+1] if d >= 0 else -d * [data[0]] + data[0:t+1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

def getStockData(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    if 'Close' not in df.columns:
        raise ValueError("Column 'Close' not found in dataset")

    return df['Close'].values

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))