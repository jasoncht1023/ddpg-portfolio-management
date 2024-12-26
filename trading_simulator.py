import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn

class TradingSimulator():

    transaction_cost = 0.005

    def __init__(self, principal, assets, start_date, end_date):
        '''
        self.data = yf.download(assets, start=start_date, end=end_date, group_by='ticker')
        if len(assets) == 1:
            # Create a MultiIndex for the columns
            multi_index_columns = pd.MultiIndex.from_tuples([(assets[0], col) for col in assets.columns])
            # Assign the new MultiIndex to the DataFrame
            self.data.columns = multi_index_columns

        returns_list = []
        # Loop through each stock ticker and calculate returns
        for stock in assets:
            # Access the 'Adj Close' prices using xs method
            adjusted_close = self.data[stock]['Adj Close']
            # Calculate percentage change
            returns_series = adjusted_close.pct_change()
            # Append the Series to the list
            returns_list.append(returns_series.rename(stock))  # Rename for clarity

        # Concatenate all return Series into a single DataFrame
        returns = pd.concat(returns_list, axis=1)

        dates = returns.index
        adj_close = self.data.xs("Adj Close", level=1, axis=1)
        adj_close = adj_close.reindex(columns=returns.columns)

        columns = pd.MultiIndex.from_product([assets, ['Adj Close', 'Returns', "MA", "RSI", "EMA_12", "EMA_26", "MACD"]])
        df = pd.DataFrame(index=dates, columns=columns)
        df.columns = columns
        for stock in assets:
            df[(stock, "Adj Close")] = adj_close[stock]
            df[(stock, "Returns")] = returns[stock]
        df = df.reset_index()

        for stock in assets:
            df[(stock, "MA")] = df[(stock, "Adj Close")].rolling(window=28).apply(self.MA)
            df.loc[0, (stock, "EMA_12")] = df.loc[0, (stock, "Adj Close")]
            df.loc[0, (stock, "EMA_26")] = df.loc[0, (stock, "Adj Close")]
            for i in range(1, len(df)):
                df.loc[i, (stock, "EMA_12")] = self.EMA(12, df.loc[i, (stock, "Adj Close")], df.loc[i-1, (stock, "EMA_12")])
                df.loc[i, (stock, "EMA_26")] = self.EMA(26, df.loc[i, (stock, "Adj Close")], df.loc[i-1, (stock, "EMA_26")])
            df[(stock, "MACD")] = df[(stock, "EMA_26")].rolling(window=9).sum() - df[(stock, "EMA_12")].rolling(window=9).sum()
            df[(stock, "RSI")] = df[(stock, "Returns")].rolling(14).apply(self.RSI)

        close_data = df.drop(df.index[:27])
        close_data = close_data.reset_index(drop=True)
        to_drop = ["EMA_12", "EMA_26", "Returns"]
        close_data = close_data.drop(columns=[(stock, label) for stock in assets for label in to_drop])

        corr = {}
        indicators = ["Adj Close", "MA", "RSI", "MACD"]
        for indicator in indicators:
            corr[indicator] = close_data.filter([(stock, indicator) for stock in assets], axis=1).corr()

        F = defaultdict(dict) # 4 * n * (m * n)
        n = len(assets)
        m = 10
        T = len(close_data) // m

        for t in range(0, T): # t
            V = close_data[t*m:(t+1)*m] # m days closing data
            for indicator in ["Adj Close", "MA", "RSI", "MACD"]: # the 4 dimensions
                for stock in assets: # n assets
                    F[t][(stock, indicator)] = V[(stock, indicator)].values.reshape(m,1).dot(corr[indicator][(stock, indicator)].values.reshape(1,n)) # m * n tensor for indicator i & stock n
        f = []
        for t in range(0, T):
            f.append([])
            for indicator in indicators:
                a = []
                for stock in tickers:
                    a.append(F[t][(stock, indicator)])
                f[-1].append(a)
        f = list(map(torch.Tensor, f))
        
        # Instantiate the network
        net = self.Conv3DNet()

        # Pass the input tensor f[t] through the network
        F_prime = []
        for t in range(0, T):
            F_prime.append(net(f[t]))
        '''

        # Stays constant throughout epochs
        self.principal = principal
        self.assets = assets

        self.data = yf.download(assets, start=start_date, end=end_date)['Adj Close']

        self.portfolio = []
        for stock in assets:
            self.portfolio.append({ "name": stock, "value": principal / len(assets), "weighting": 1 / len(assets) })

        self.portfolio_value = principal
        self.time = 0
    
    # Define the 3D Convolutional Neural Network layer
    class Conv3DNet(nn.Module):
        def __init__(self):
            super(self.Conv3DNet, self).__init__()
            self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv3d(x)
            x = self.relu(x)
            return x
    
    def EMA(self, w, price, last):
        a = 2/(1+w)
        return a*price + (1-a)*last
    
    def MA(self, prices):
        return sum(prices) / 28
    
    def MACD(self, long, short):
        return sum(long) - sum(short)
    
    def RSI(self, returns):
        avg_gain = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        return 100 * (1 - 1/(1+avg_gain/avg_loss))
        
    def restart(self):
        self.portfolio = []
        for stock in self.assets:
            self.portfolio.append({"name": stock, 
                                   "value": self.principal / len(self.assets), 
                                   "weighting": 1 / len(self.assets), 
                                   "price": self.data.iloc[0][stock],
                                   "num_shares": self.principal / len(self.assets) / self.data.iloc[0][stock]})
        self.portfolio_value = self.principal
        self.time = 0

        new_state = []
        for stock in self.portfolio:
            new_state.extend([stock["value"], stock["weighting"], stock["price"], stock["num_shares"]])
        new_state.extend(self.data.iloc[0])

        return new_state

    def step(self, action):
        done = 0
        self.time += 1
        old_portfolio_value = self.portfolio_value

        new_value = 0
        for i in range(len(self.portfolio)):
            weight_adjusted_value = old_portfolio_value * action[i]
            self.portfolio[i]["weighting"] = action[i]
            self.portfolio[i]["num_shares"] = weight_adjusted_value / self.portfolio[i]["price"]

            # Price and value of a particular stock change in time = t+1
            self.portfolio[i]["price"] = self.data.iloc[self.time][self.assets[i]]
            self.portfolio[i]["value"] = self.portfolio[i]["price"] * self.portfolio[i]["num_shares"]
            new_value +=  self.portfolio[i]["value"]
        self.portfolio_value = new_value

        print(self.time)
        if (self.time == len(self.data)-1):
            done = 1

        reward = self.portfolio_value - old_portfolio_value

        new_state = []
        for stock in self.portfolio:
            new_state.extend([stock["value"], stock["weighting"], stock["price"], stock["num_shares"]])
        new_state.extend(self.data.iloc[self.time])

        return new_state, reward, done

