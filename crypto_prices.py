import requests
import time

def get_prices(coins):
    ids = ",".join(coins)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data

coins = ["bitcoin", "ethereum", "solana"]
prices = get_prices(coins)

for coin in coins:
    if coin in prices:
        price = prices[coin]["usd"]
        print(f"{coin.upper()}: ${price:,}")
    else:
        print(f"{coin.upper()}: unavailable")