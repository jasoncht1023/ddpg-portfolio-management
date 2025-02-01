import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import copy
from datetime import datetime, timedelta

class TradingSimulator:
    def __init__(self, principal, assets, start_date, end_date, rebalance_window, tx_fee_per_share):
        compute_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=10)
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
        returns = returns[start_index:].set_index('Date')

        dates = returns.index
        adj_close = self.data.xs("Adj Close", level=1, axis=1)
        adj_close = adj_close.reindex(columns=returns.columns)

        columns = pd.MultiIndex.from_product([assets, ['Adj Close', 'Returns', "RSI",]])
        df = pd.DataFrame(index=dates, columns=columns)
        df.columns = columns
        for stock in assets:
            df[(stock, "Adj Close")] = adj_close[stock]
            df[(stock, "Returns")] = returns[stock]
        df = df.reset_index()

        for stock in assets:
            df[(stock, "RSI")] = df[(stock, "Returns")].rolling(2).apply(self.__RSI)

        close_data = df[1:len(df) - (len(df)-1)%10]
        close_data = close_data.reset_index(drop=True)
        to_drop = ["Returns"]
        close_data = close_data.drop(columns=[(stock, label) for stock in assets for label in to_drop])

        self.trading_dates = close_data["Date"].dt.date.astype(str).tolist()[1:]

        # Collect all the Adjusted Close price data
        adj_close_data = close_data.loc[:, close_data.columns.get_level_values(1) == "Adj Close"]
        adj_close_data.columns = adj_close_data.columns.droplevel(1)
        self.close_price = adj_close_data

        rsi_data = close_data.loc[:, close_data.columns.get_level_values(1) == "RSI"]
        rsi_data.columns = rsi_data.columns.droplevel(1)
        self.rsi = rsi_data

        # Stays constant throughout epochs
        self.principal = principal
        self.assets = assets
        self.rebalance_window = rebalance_window
        self.tx_fee = tx_fee_per_share

    def __RSI(self, returns):
        if (len(returns[returns > 0]) == 0):
            return 0
        if (len(returns[returns < 0]) == 0):
            return 100
        avg_gain = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        return 100 * (1 - 1/(1+avg_gain/avg_loss))
    
    def sharpe_ratio(self):     
        # Load the risk free rates and convert to 1-window-days rate
        risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        risk_free_rates.columns = ["date", "risk_free_rate"]
        risk_free_rates["risk_free_rate"] = risk_free_rates["risk_free_rate"] * self.rebalance_window / 365

        # Compute the return rates
        value_pct_change = pd.Series(self.value_history).pct_change() * 100
        return_rates = pd.DataFrame({"date": self.trading_dates[1:], "return": value_pct_change[1:]})
        return_rates.columns = ["date", "return_rate"]

        # Combine the risk free rates and return rates
        df = return_rates.set_index('date').join(risk_free_rates.set_index('date'))

        # Fill missing risk free rates
        df["risk_free_rate"] = df["risk_free_rate"].interpolate().bfill().ffill()

        df["excess_return_rate"] = df["return_rate"] - df["risk_free_rate"]

        # print(df)
        return df["excess_return_rate"].mean() / df["excess_return_rate"].std()
    
    def total_portfolio_value(self):
        return self.portfolio_value

    def restart(self):
        # Reset the initial portfolio
        self.portfolio = []
        self.portfolio_value = self.principal
        self.time = 1

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
        curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                      # Close price of each asset at t
        prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                    # Close price of each asset at t-1
        log_return = np.log(np.divide(curr_close_price, prev_close_price))                              # Natural log of return
        rsi = [x for x in self.rsi.iloc[self.time]]                                                     # RSI of each asset at time t
        rsi = rsi / 100
        holdings = [asset["weighting"] for asset in self.portfolio]                                     # Share and cash weightings 
        curr_close_price = (curr_close_price - curr_close_price.mean())/(curr_close_price.std())        # Normalize closing price
        prev_close_price = (prev_close_price - prev_close_price.mean())/(prev_close_price.std())

        initial_input = np.concatenate((curr_close_price, prev_close_price, log_return, rsi, holdings, [np.log(self.portfolio_value / self.principal)]))

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
                self.portfolio[i]["price"] = self.close_price.iloc[self.time][self.assets[i]]
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

        self.portfolio_value = new_value
        self.value_history.append(new_value)
        # reward = np.log(self.portfolio_value / old_portfolio_value)
        reward = self.portfolio_value - old_portfolio_value - total_tx_cost
        # reward = self.portfolio_value - old_portfolio_value

        # New states
        curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                    # Close price of each asset at t
        prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                  # Close price of each asset at t-1
        log_return = np.array(np.log(np.divide(curr_close_price, prev_close_price)))                  # Natural log of return
        rsi = np.array([x for x in self.rsi.iloc[self.time]])                                         # RSI of each asset at time t
        rsi = rsi/100
        holdings = np.array([asset["weighting"] for asset in self.portfolio])                         # Share and cash holdings 
        curr_close_price = (curr_close_price - curr_close_price.mean())/(curr_close_price.std())      # Normalize closing price
        prev_close_price = (prev_close_price - prev_close_price.mean())/(prev_close_price.std())

        new_state = np.concatenate((curr_close_price, prev_close_price, log_return, rsi, holdings, [np.log(self.portfolio_value / self.principal)]))

        if (self.time == len(self.close_price)-1):                                                    # Indicate the end of the episode 
            done = 1

        return new_state, reward, done
