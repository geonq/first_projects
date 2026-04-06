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
