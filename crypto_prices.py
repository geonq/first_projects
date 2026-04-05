import requests

def get_price(coin):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    print(data)  # add this line
    return data[coin]["usd"]

coins = ["bitcoin", "ethereum", "solana"]

for coin in coins:
    price = get_price(coin)
    print(f"{coin.upper()}: ${price:,}")