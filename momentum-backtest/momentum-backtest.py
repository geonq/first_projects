import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from io import StringIO

response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
df = pd.read_csv(StringIO(response.text)) 
na_values=["."]
trading_days: int = 252
momentum_windows = [7, 14, 30]
default_cost_bps = 5

df.tail()

