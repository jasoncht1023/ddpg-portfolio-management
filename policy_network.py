import torch 
import torch.nn as nn
import torch.optim as optim
import tensorly as tl
from tensorly.decomposition import tucker
from collections import defaultdict
import pandas as pd
import yfinance as yf
import numpy as np
from functools import reduce 
import operator
class ActorNetwork(nn.Module):
    def __init__(self, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.tucker_dimension = [8, 6, 6, 6]
        self.num_of_actions = 10
        self.relu = nn.ReLU()
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1,3,1))
        self.fc = nn.Linear(reduce(operator.mul, self.tucker_dimension, 1), self.num_of_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
    def forward(self, x):
        x = self.conv3d(x)
        x = self.relu(x)
        x = tl.tensor(x.detach().cpu().numpy())
        core, factors = tucker(x, rank=self.tucker_dimension) # can be change
        x = torch.tensor(core)
        x = torch.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

def EMA(w, price, last):
    a = 2/(1+w)
    return a*price + (1-a)*last
def MA(prices):
    return sum(prices) / 28
def MACD(long, short):
    return sum(long) - sum(short)
def RSI(returns):
    avg_gain = returns[returns > 0].mean()
    avg_loss = -returns[returns < 0].mean()
    return 100 * (1 - 1/(1+avg_gain/avg_loss))
    
# for testing only
if __name__ == "__main__":
    stock_list = ["MSFT","AAPL","GOOG","MMM","GS","NKE","AXP","HON","CRM","JPM"]
    data = yf.download(stock_list, start="2024-01-01", end="2024-11-11", group_by='ticker')
    if len(stock_list) == 1:
        multi_index_columns = pd.MultiIndex.from_tuples([(tickers[0], col) for col in data.columns])
        data.columns = multi_index_columns
    returns_list = []
    for stock in stock_list:
        adjusted_close = data[stock]['Adj Close']
        returns_series = adjusted_close.pct_change()
        returns_list.append(returns_series.rename(stock))  # Rename for clarity
    returns = pd.concat(returns_list, axis=1)
    dates = returns.index
    adj_close = data.xs("Adj Close", level=1, axis=1)
    adj_close = adj_close.reindex(columns=returns.columns)
    columns = pd.MultiIndex.from_product([stock_list, ['Adj Close', 'Returns', "MA", "RSI", "EMA_12", "EMA_26", "MACD"]])
    df = pd.DataFrame(index=dates, columns=columns)
    df.columns = columns
    for stock in stock_list:
        df[(stock, "Adj Close")] = adj_close[stock]
        df[(stock, "Returns")] = returns[stock]
    df = df.reset_index()
    for stock in stock_list:
        df[(stock, "MA")] = df[(stock, "Adj Close")].rolling(window=28).apply(MA)
    
        df.loc[0, (stock, "EMA_12")] = df.loc[0, (stock, "Adj Close")]
        df.loc[0, (stock, "EMA_26")] = df.loc[0, (stock, "Adj Close")]
        for i in range(1, len(df)):
            df.loc[i, (stock, "EMA_12")] = EMA(12, df.loc[i, (stock, "Adj Close")], df.loc[i-1, (stock, "EMA_12")])
            df.loc[i, (stock, "EMA_26")] = EMA(26, df.loc[i, (stock, "Adj Close")], df.loc[i-1, (stock, "EMA_26")])
    
        df[(stock, "MACD")] = df[(stock, "EMA_26")].rolling(window=9).sum() - df[(stock, "EMA_12")].rolling(window=9).sum()
        df[(stock, "RSI")] = df[(stock, "Returns")].rolling(14).apply(RSI)
    close_data = df.drop(df.index[:27])
    close_data = close_data.reset_index(drop=True)
    to_drop = ["EMA_12", "EMA_26", "Returns"]
    close_data = close_data.drop(columns=[(stock, label) for stock in stock_list for label in to_drop])
    corr = {}
    indicators = ["Adj Close", "MA", "RSI", "MACD"]
    for indicator in indicators:
        corr[indicator] = close_data.filter([(stock, indicator) for stock in stock_list], axis=1).corr()
    F = defaultdict(dict)
    n = len(stock_list)
    m = 10
    T = len(close_data) // m
    
    for t in range(0, T): # t
        V = close_data[t*m:(t+1)*m] # m days closing data
        for indicator in ["Adj Close", "MA", "RSI", "MACD"]: # the 4 dimensions
            for stock in stock_list: # n assets
                F[t][(stock, indicator)] = V[(stock, indicator)].values.reshape(m,1).dot(corr[indicator][(stock, indicator)].values.reshape(1,n)) # m * n tensor for indicator i & stock n
    f = []
    for t in range(0, T):
        f.append([])
        for indicator in indicators:
            a = []
            for stock in stock_list:
                a.append(F[t][(stock, indicator)])
            f[-1].append(a)
    f = list(map(torch.Tensor, f))
    print(f"before: {f[0].shape}")
    policy_net = ActorNetwork("model name")
    f_prime = policy_net(f[0])
    print(f"after: {f_prime.shape}")
    print(f_prime)
    print(f"sum: {sum(f_prime)}")