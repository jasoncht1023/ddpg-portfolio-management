import yfinance as yf

data = yf.download("MSFT AAPL GOOG MMM GS NKE AXP HON CRM JPM", period="10y", interval="1d")
data.to_csv("stock_new_fetch.csv")