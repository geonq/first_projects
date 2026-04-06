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

    def run(self):
        self.generate_signals()
        self.calculate_returns()
        self.results = self.calculate_metrics()
        return self.results

    def print_results(self):
        if self.results is None:
            print("Run the backtest first to see results.")
            return
        print(f"Total Return: {self.results['total_return']:.2%}")
        print(f"Annualized Return: {self.results['annualized_return']:.2%}")
        print(f"Volatility: {self.results['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Position Changes: {self.results['position_changes']}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Buy and Hold Total Return: {self.data['cumulative_buyhold'].iloc[-1] - 1:.2%}")

    def plot_results(self):
        # Chart 1: Equity curves
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data["cumulative_strategy"], label="Momentum Strategy")
        plt.plot(self.data.index, self.data["cumulative_buyhold"], label="Buy and Hold")
        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid()
        plt.show()

        # Chart 2: Drawdown
        running_max = self.data["cumulative_strategy"].cummax()
        drawdown = (self.data["cumulative_strategy"] - running_max) / running_max
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, drawdown, label="Drawdown", color="red")
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid()
        plt.show()

        # Chart 3: Position
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data["position"], label="Position", color="green")
        plt.title("Position Over Time")
        plt.xlabel("Date")
        plt.ylabel("Position (1=Long, 0=Cash)")
        plt.legend()
        plt.grid()
        plt.show()

        

if __name__ == "__main__":
    risk_free_rate = fetch_fred_rate()
    qqq_data = fetch_qqq_data()
    bt = MomentumBacktest(qqq_data, risk_free_rate)
    bt.run()
    bt.print_results()
    bt.plot_results()

