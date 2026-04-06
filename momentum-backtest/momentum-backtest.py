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
    return risk_free_rate

def fetch_qqq_data():
    df = yf.download("QQQ", period="10y")
    if df is None or df.empty:
        raise ValueError("Failed to fetch data for QQQ.")
    df.columns = df.columns.get_level_values(0)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    return df

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

    def calculate_returns(self):
        self.data["strategy_log_return"] = self.data["position"] * self.data["log_return"]
        self.data["cost"] = self.data["position_change"] * self.transaction_cost_bps
        self.data["strategy_net_log_return"] = self.data["strategy_log_return"] - self.data["cost"]
        self.data["cumulative_strategy"] = self.data["strategy_net_log_return"].cumsum().apply(np.exp)
        self.data["cumulative_buyhold"] = self.data["log_return"].cumsum().apply(np.exp)
    
    def calculate_metrics(self):
        total = self.data["strategy_net_log_return"].sum()
        years = len(self.data) / trading_days
        std_vol = self.data["strategy_net_log_return"].std() * np.sqrt(trading_days)
        sharpe = (total / years - self.risk_free_rate) / std_vol
        running_max = self.data["cumulative_strategy"].cummax()
        drawdown = (self.data["cumulative_strategy"] - running_max) / running_max
        max_drawdown = drawdown.min()
        position_changes = self.data["position_change"].sum()
        win_rate = (self.data[self.data["position"] == 1]["log_return"] > 0).mean()
        return {
            "total_return": total,
            "annualized_return": total / years,
            "volatility": std_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "position_changes": position_changes,
            "win_rate": win_rate
        }


if __name__ == "__main__":
    risk_free_rate = fetch_fred_rate()
    qqq_data = fetch_qqq_data()
    print(qqq_data.tail())
