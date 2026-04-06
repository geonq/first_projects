import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from io import StringIO

trading_days: int = 252
momentum_windows = [7, 14, 30]
default_cost_bps = 5

def fetch_fred_rate():
    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
    df = pd.read_csv(StringIO(response.text), na_values=["."])
    print(df.tail())
    df = df.dropna()
    risk_free_rate = df["DGS10"].iloc[-1].item() / 100
    print(f"Risk-free rate: {risk_free_rate}")
    fred_rate_change = df["DGS10"].iloc[-1].item() - df["DGS10"].iloc[-2].item()
    if fred_rate_change > 0:
        print(f"The risk-free rate has increased {abs(fred_rate_change):.2f}% since the last data point.")
    else:
        print(f"The risk-free rate has decreased {abs(fred_rate_change):.2f}% since the last data point.")
fetch_fred_rate()

def fetch_qqq_data():
    df = yf.download("QQQ", period="10y")
    if df is None or df.empty:
        raise ValueError("Failed to fetch data for QQQ.")
    df.columns = df.columns.get_level_values(0)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    return df
print(fetch_qqq_data().tail())

class MomentumBacktest:
    def __init__(self, data, risk_free_rate, transaction_cost_bps=default_cost_bps, momentum_windows=momentum_windows):
        self.data = data.copy()
        self.risk_free_rate = risk_free_rate
        self.momentum_windows = momentum_windows
        self.transaction_cost_bps = transaction_cost_bps / 10_000
        self.results = None
    
    def generate_signals(self):
        for window in self.momentum_windows:
            self.data[f"mom_{window}"] = self.data["log_return"].rolling(window=window).sum()
        self.data["position"]=((self.data["mom_7"]> 0) & (self.data["mom_14"]>0) & (self.data["mom_30"]>0)).astype(int)
        self.data["position"] = self.data["position"].shift(1).fillna(0)
        self.data["position_change"] = self.data["position"].diff().abs()
