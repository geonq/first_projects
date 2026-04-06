import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from io import StringIO

trading_days: int = 252

def fetch_fred_rate():
    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO")
    df = pd.read_csv(StringIO(response.text), na_values=["."])
    df = df.dropna()
    risk_free_rate = df["DGS3MO"].iloc[-1].item() / 100
    print(f"Risk-free rate: {risk_free_rate}")
    fred_rate_change = df["DGS3MO"].iloc[-1].item() - df["DGS3MO"].iloc[-2].item()
    last_date = df["observation_date"].iloc[-1]
    if fred_rate_change > 0:
        print(f"The risk-free rate has increased {abs(fred_rate_change):.2f}% since {last_date}.")
    else:
        print(f"The risk-free rate has decreased {abs(fred_rate_change):.2f}% since {last_date}.")
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
    def __init__(self, data, risk_free_rate, transaction_cost_bps: float = 5, momentum_windows=None):
        self.momentum_windows = momentum_windows or [7, 14, 30]
        self.data = data.copy()
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_bps = transaction_cost_bps / 10_000
        self.results: dict | None = None
    
    def generate_signals(self, strategy="baseline"):
        for window in self.momentum_windows:
            self.data[f"mom_{window}"] = (
                self.data["log_return"].rolling(window=window).sum()
            )

        if strategy == "baseline":
            signal = pd.concat(
                [self.data[f"mom_{w}"] > 0 for w in self.momentum_windows],
                axis=1
            ).all(axis=1).astype(int)

        elif strategy == "majority":
            signal = (
                pd.concat(
                    [self.data[f"mom_{w}"] > 0 for w in self.momentum_windows],
                    axis=1
                ).sum(axis=1) >= 2
            ).astype(int)

        elif strategy == "ma200":
            self.data["ma200"] = self.data["Close"].rolling(200).mean()
            signal = (
                (self.data["mom_30"] > 0) &
                (self.data["Close"] > self.data["ma200"])
            ).astype(int)

        elif strategy == "long_windows":
            for w in [30, 60, 90]:
                self.data[f"mom_{w}"] = (
                   self.data["log_return"].rolling(window=w).sum()
                )
            signal = pd.concat(
                [self.data[f"mom_{w}"] > 0 for w in [30, 60, 90]],
                axis=1
            ).all(axis=1).astype(int)

        self.data["position"] = signal.shift(1).fillna(0)
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
        annualized_return = np.exp(total / years) - 1
        sharpe = (annualized_return - self.risk_free_rate) / std_vol
        running_max = self.data["cumulative_strategy"].cummax()
        drawdown = (self.data["cumulative_strategy"] - running_max) / running_max
        max_drawdown = drawdown.min()
        position_changes = self.data["position_change"].sum()
        win_rate = (self.data[self.data["position"] == 1]["strategy_net_log_return"] > 0).mean()
        self.data["drawdown"] = drawdown
        return {
                "total_return": np.exp(total) - 1,
                "annualized_return": np.exp(total / years) - 1,
                "volatility": std_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "position_changes": position_changes,
                "win_rate": win_rate
            }

    def run(self, strategy="baseline"):
        self.generate_signals(strategy=strategy)
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
        print(f"Buy and Hold Total Return: {self.data['cumulative_buyhold'].iloc[-1].item() - 1:.2%}")

    def plot_results(self):
        if self.results is None:
            print("Run the backtest first.")
            return
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
        drawdown = self.data["drawdown"]
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

class MonteCarloValidator:
    def __init__(self, backtest: MomentumBacktest, n_simulations=10_000):
        self.backtest = backtest
        self.n_simulations = n_simulations
        self.simulated_sharpes: np.ndarray = np.array([])

    def run(self):
        assert self.backtest.results is not None, "Run the backtest first."
        log_returns = self.backtest.data["log_return"].values
        actual_sharpe = self.backtest.results["sharpe_ratio"]
        sharpes: list[float] = []

        for _ in range(self.n_simulations):
            shuffled = np.random.permutation(log_returns)
            shuffled_series = pd.Series(
                shuffled,
                index=self.backtest.data.index
            )
            sim_df = self.backtest.data.copy()
            sim_df["log_return"] = shuffled_series

            sim_bt = MomentumBacktest(
                sim_df[["log_return"]],
                self.backtest.risk_free_rate,
                self.backtest.transaction_cost_bps * 10_000
            )
            sim_bt.run()
            assert sim_bt.results is not None
            sharpes.append(sim_bt.results["sharpe_ratio"])

        self.simulated_sharpes = np.array(sharpes)
        p_value = (self.simulated_sharpes >= actual_sharpe).mean()
        return p_value

    def print_results(self):
        assert self.backtest.results is not None, "Run the backtest first."
        actual_sharpe = self.backtest.results["sharpe_ratio"]
        p_value = (self.simulated_sharpes >= actual_sharpe).mean()
        percentile = (self.simulated_sharpes < actual_sharpe).mean() * 100

        print(f"\n--- Monte Carlo Validation ({self.n_simulations:,} simulations) ---")
        print(f"Actual Sharpe:     {actual_sharpe:.3f}")
        print(f"Simulated mean:    {self.simulated_sharpes.mean():.3f}")
        print(f"Simulated std:     {self.simulated_sharpes.std():.3f}")
        print(f"Percentile rank:   {percentile:.1f}th")
        print(f"p-value:           {p_value:.4f}")
        if p_value < 0.05:
            print("RESULT: Strategy shows statistically significant edge (p < 0.05)")
        else:
            print("RESULT: Cannot reject null hypothesis — may be noise")

    def plot_results(self):
        assert self.backtest.results is not None, "Run the backtest first."
        actual_sharpe = self.backtest.results["sharpe_ratio"]
        plt.figure(figsize=(10, 5))
        plt.hist(self.simulated_sharpes, bins=100, color="steelblue",
                 edgecolor="white", alpha=0.8, label="Shuffled (random)")
        plt.axvline(actual_sharpe, color="red", linewidth=2,
                    label=f"Actual Sharpe: {actual_sharpe:.3f}")
        plt.title("Monte Carlo Validation — Is the Edge Real?")
        plt.xlabel("Sharpe Ratio")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__":
    try:
        risk_free_rate = fetch_fred_rate()
    except requests.exceptions.RequestException:
        print("FRED unavailable, defaulting to 4%")
        risk_free_rate = 0.04

    qqq_data = fetch_qqq_data()

    strategies = ["baseline", "majority", "ma200", "long_windows"]
    summary = []

    for s in strategies:
        bt = MomentumBacktest(qqq_data.copy(), risk_free_rate)
        bt.run(strategy=s)
        assert bt.results is not None

        mc = MonteCarloValidator(bt, n_simulations=1000)
        mc.run()

        p_value = (mc.simulated_sharpes >= bt.results["sharpe_ratio"]).mean()

        summary.append({
            "strategy":         s,
            "sharpe":           round(bt.results["sharpe_ratio"], 3),
            "ann_return":       round(bt.results["annualized_return"], 4),
            "max_drawdown":     round(bt.results["max_drawdown"], 4),
            "position_changes": bt.results["position_changes"],
            "p_value":          round(p_value, 4),
        })
        print(f"\n=== {s.upper()} ===")
        bt.print_results()
        print(f"p-value: {p_value:.4f}")

    print("\n\n=== COMPARISON TABLE ===")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

