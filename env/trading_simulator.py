import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import copy
from datetime import datetime, timedelta

class TradingSimulator:
    def __init__(self, principal, assets, start_date, end_date, rebalance_window, tx_fee_per_share):
        compute_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=50)
        compute_date = compute_date.strftime('%Y-%m-%d')

        self.data = yf.download(assets, start=compute_date, end=end_date, group_by="ticker")
        if len(assets) == 1:
            # Create a MultiIndex for the columns
            multi_index_columns = pd.MultiIndex.from_tuples([(assets[0], col) for col in assets.columns])
            # Assign the new MultiIndex to the DataFrame
            self.data.columns = multi_index_columns

        returns_list = []
        # Loop through each stock ticker and calculate returns
        for stock in assets:
            # Access the 'Adj Close' prices using xs method
            adjusted_close = self.data[stock]["Adj Close"]
            # Calculate percentage change
            returns_series = adjusted_close.pct_change()
            # Append the Series to the list
            returns_list.append(returns_series.rename(stock))  # Rename for clarity

        # Concatenate all return Series into a single DataFrame
        returns = pd.concat(returns_list, axis=1)

        # Get 27 days data before start date for indicators computation
        returns.reset_index(inplace=True)
        start_index = returns[returns['Date'] >= start_date].index[0]
        returns = returns[start_index-27:].set_index('Date')

        dates = returns.index
        adj_close = self.data.xs("Adj Close", level=1, axis=1)
        adj_close = adj_close.reindex(columns=returns.columns)

        columns = pd.MultiIndex.from_product([assets, ["Adj Close", "Returns", "MA", "RSI", "EMA_12", "EMA_26", "MACD"]])
        df = pd.DataFrame(index=dates, columns=columns)
        df.columns = columns
        for stock in assets:
            df[(stock, "Adj Close")] = adj_close[stock]
            df[(stock, "Returns")] = returns[stock]
        df = df.reset_index()

        for stock in assets:
            df[(stock, "MA")] = df[(stock, "Adj Close")].rolling(window=28).apply(self.__MA)
            df.loc[0, (stock, "EMA_12")] = df.loc[0, (stock, "Adj Close")]
            df.loc[0, (stock, "EMA_26")] = df.loc[0, (stock, "Adj Close")]
            for i in range(1, len(df)):
                df.loc[i, (stock, "EMA_12")] = self.__EMA(12, df.loc[i, (stock, "Adj Close")], df.loc[i - 1, (stock, "EMA_12")])
                df.loc[i, (stock, "EMA_26")] = self.__EMA(26, df.loc[i, (stock, "Adj Close")], df.loc[i - 1, (stock, "EMA_26")])
            df[(stock, "MACD")] = df[(stock, "EMA_26")].rolling(window=9).sum() - df[(stock, "EMA_12")].rolling(window=9).sum()
            df[(stock, "RSI")] = df[(stock, "Returns")].rolling(14).apply(self.__RSI)

        close_data = df[27:len(df) - (len(df)-27)%10]
        close_data = close_data.reset_index(drop=True)
        to_drop = ["EMA_12", "EMA_26", "Returns"]
        close_data = close_data.drop(columns=[(stock, label) for stock in assets for label in to_drop])

        trading_dates = close_data["Date"].dt.date.astype(str).tolist()
        self.rebalance_dates = [trading_dates[i] for i in range(len(trading_dates)) if (i+1) % rebalance_window == 0]

        # Collect all the Adjusted Close price data
        adj_close_data = close_data.loc[:, close_data.columns.get_level_values(1) == "Adj Close"]
        adj_close_data.columns = adj_close_data.columns.droplevel(1)
        self.close_price = adj_close_data

        corr = {}
        indicators = ["Adj Close", "MA", "RSI", "MACD"]
        for indicator in indicators:
            corr[indicator] = close_data.filter([(stock, indicator) for stock in assets], axis=1).corr()

        F = defaultdict(dict)  # 4 * n * (m * n)
        n = len(assets)
        T = len(close_data) // rebalance_window

        for t in range(0, T):  # t
            V = close_data[t * rebalance_window : (t + 1) * rebalance_window]  # m days closing data
            for indicator in ["Adj Close", "MA", "RSI", "MACD"]:  # the 4 dimensions
                for stock in assets:  # n assets
                    # m * n tensor for indicator i & stock n
                    F[t][(stock, indicator)] = V[(stock, indicator)].values.reshape(rebalance_window, 1).dot(corr[indicator][(stock, indicator)].values.reshape(1, n))

        f = []
        for t in range(0, T):
            f.append([])
            for indicator in indicators:
                a = []
                for stock in assets:
                    a.append(F[t][(stock, indicator)])
                f[-1].append(a)
        f = list(map(torch.Tensor, f))
        self.stock_tensors = f

        # Stays constant throughout epochs
        self.principal = principal
        self.assets = assets
        self.rebalance_window = rebalance_window
        self.tx_fee = tx_fee_per_share

    def __EMA(self, w, price, last):
        a = 2 / (1 + w)
        return a * price + (1 - a) * last

    def __MA(self, prices):
        return sum(prices) / 28

    def __MACD(self, long, short):
        return sum(long) - sum(short)

    def __RSI(self, returns):
        avg_gain = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        return 100 * (1 - 1 / (1 + avg_gain / avg_loss))
    
    def sharpe_ratio(self):     
        # Load the risk free rates and convert to 1-window-days rate
        risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        risk_free_rates.columns = ["date", "risk_free_rate"]
        risk_free_rates["risk_free_rate"] = risk_free_rates["risk_free_rate"] * self.rebalance_window / 365

        # Compute the return rates
        value_pct_change = pd.Series(self.value_history).pct_change() * 100
        return_rates = pd.DataFrame({"date": self.rebalance_dates, "return": value_pct_change[1:]})
        return_rates.columns = ["date", "return_rate"]

        # Combine the risk free rates and return rates
        df = return_rates.set_index('date').join(risk_free_rates.set_index('date'))

        # Fill missing risk free rates
        df["risk_free_rate"] = df["risk_free_rate"].interpolate().bfill().ffill()

        df["excess_return_rate"] = df["return_rate"] - df["risk_free_rate"]

        print(df)
        return df["excess_return_rate"].mean() / df["excess_return_rate"].std()  

    def restart(self):
        # Reset the initial portfolio
        self.portfolio = []
        self.portfolio_value = self.principal
        self.time = 0

        # Portfolio value history for return computation used in evaluation
        self.value_history = [self.principal]

        # Initializing the stock part of the portfolio
        for stock in self.assets:
            self.portfolio.append({
                "name": stock,
                "value": 0,
                "weighting": 0,
                "price": self.close_price.iloc[0][stock],
                "num_shares": 0
            })
        
        # Initially only cash is held in the portfolio
        self.portfolio.append({
            "name": "cash",
            "value": self.principal,
            "weighting": 1,
            "price": 1,
            "num_shares": self.principal
        })

        # Initial observation
        initial_input = self.stock_tensors[0]

        return initial_input

    def step(self, action):
        done = 0
        self.time += 1
        old_portfolio_value = self.portfolio_value
        old_portfolio = copy.deepcopy(self.portfolio)

        # print("Time step:", self.time)

        # Compute the new portfolio value after 1 rebalance window
        # Price and value of a particular stock change in time = t+1, price of cash is unchanged
        new_value = 0
        for i in range(len(self.portfolio)):
            if (self.portfolio[i]["name"] != "cash"):
                self.portfolio[i]["price"] = self.close_price.iloc[self.time * self.rebalance_window - 1][self.assets[i]]
            self.portfolio[i]["value"] = self.portfolio[i]["price"] * self.portfolio[i]["num_shares"]
            new_value += self.portfolio[i]["value"]
        
        # Adjust the weighting of each asset in the portfolio based on the new portfolio value
        # An empty action array means skipping the portfolio rebalance, not applicable in RL algorithms
        if (len(action) != 0):
            for i in range(len(self.portfolio)):
                weight_adjusted_stock_value = new_value * action[i]
                self.portfolio[i]["weighting"] = action[i]
                self.portfolio[i]["num_shares"] = weight_adjusted_stock_value / self.portfolio[i]["price"]
                self.portfolio[i]["value"] = weight_adjusted_stock_value
        else:
            for i in range(len(self.portfolio)):
                self.portfolio[i]["weighting"] = self.portfolio[i]["value"] / new_value
        
        # Compute the transaction fee based on the number of shares bought/sold
        total_tx_cost = 0
        for i in range(len(self.portfolio)-1):
            total_tx_cost += abs(self.portfolio[i]["num_shares"] - old_portfolio[i]["num_shares"]) * self.tx_fee
        # print("transaction_cost:", total_tx_cost)
        # print()

        self.portfolio_value = new_value
        self.value_history.append(new_value)
        # reward = np.log((self.portfolio_value - old_portfolio_value - total_tx_cost) / old_portfolio_value)
        reward = self.portfolio_value - old_portfolio_value - total_tx_cost

        if (self.time == np.ceil(len(self.close_price) / self.rebalance_window)):           # The episode is ended
            done = 1
            new_state = np.zeros((4, len(self.assets), self.rebalance_window, len(self.assets)))
        else:                                                                               # The episode is not ended
            new_state = self.stock_tensors[self.time]

        return new_state, reward, done
