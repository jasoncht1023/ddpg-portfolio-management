import yfinance as yf
import pandas as pd
import numpy as np
import copy
from datetime import datetime, timedelta
from env.asset import Asset

class TradingSimulator:
    def __init__(self, principal, assets, start_date, end_date, rebalance_window, tx_fee_per_share):
        compute_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=10)
        compute_date = compute_date.strftime('%Y-%m-%d')

        self.data = yf.download(assets, start=compute_date, end=end_date, group_by="ticker", auto_adjust=True)
        if len(assets) == 1:
            # Create a MultiIndex for the columns
            multi_index_columns = pd.MultiIndex.from_tuples([(assets[0], col) for col in assets.columns])
            # Assign the new MultiIndex to the DataFrame
            self.data.columns = multi_index_columns

        returns_list = []
        # Loop through each stock ticker and calculate returns
        for stock in assets:
            # Access the 'Close' prices using xs method
            adjusted_close = self.data[stock]["Close"]
            # Calculate percentage change
            returns_series = adjusted_close.pct_change()
            # Append the Series to the list
            returns_list.append(returns_series.rename(stock))  # Rename for clarity

        # Concatenate all return Series into a single DataFrame
        returns = pd.concat(returns_list, axis=1)

        # Get 27 days data before start date for indicators computation
        returns.reset_index(inplace=True)
        returns = returns.set_index('Date')

        dates = returns.index
        adj_close = self.data.xs("Close", level=1, axis=1)
        adj_close = adj_close.reindex(columns=returns.columns)

        columns = pd.MultiIndex.from_product([assets, ['Close', 'Returns', "RSI"]])
        df = pd.DataFrame(index=dates, columns=columns)
        df.columns = columns
        for stock in assets:
            df[(stock, "Close")] = adj_close[stock]
            df[(stock, "Returns")] = returns[stock]
        df = df.reset_index()

        for stock in assets:
            df[(stock, "RSI")] = df[(stock, "Returns")].rolling(2).apply(self.__RSI)

        close_data = df[1:len(df)]                                                                          # Drop the first row without the RSI value
        close_data = close_data.reset_index(drop=True)
        to_drop = ["Returns"]
        close_data = close_data.drop(columns=[(stock, label) for stock in assets for label in to_drop])     # Drop unused columns
        start_index = close_data[close_data['Date'] >= start_date].index[0]                                 # Index of the first trading date in range
        close_data = close_data[start_index-1:].reset_index(drop=True)                                      # Take an extra day before the first in-range trading date

        self.trading_dates = close_data["Date"].dt.date.astype(str).tolist()[1:]

        # Collect all the Adjusted Close price data
        adj_close_data = close_data.loc[:, close_data.columns.get_level_values(1) == "Close"]
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

    # Compute the rate of return of the whole period
    def period_return(self, periood_return_history):
        start_value = periood_return_history.iloc[0] 
        end_value = periood_return_history.iloc[-1] 
        return (end_value - start_value) / start_value * 100 
    
    def sharpe_ratio(self):      
        # Load the annual risk free rates
        risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        risk_free_rates['date'] = pd.to_datetime(risk_free_rates['date']).dt.year
        risk_free_rates.columns = ["year", "risk_free_rate"]

        # Compute the return rates
        portfolio_history = pd.DataFrame({"date": pd.to_datetime(self.trading_dates), "value": self.value_history})
        annual_return_rates = portfolio_history.groupby(portfolio_history['date'].dt.year)['value'].apply(self.period_return).reset_index()
        annual_return_rates.columns = ["year", "return_rate"]

        # Combine the risk free rates and return rates
        df = annual_return_rates.set_index('year').join(risk_free_rates.set_index('year'))

        df["excess_return_rate"] = df["return_rate"] - df["risk_free_rate"]

        # print(df)
        return df["excess_return_rate"].mean() / df["excess_return_rate"].std()
    
    def omega_ratio(self, target_rate):
        window_target_rate = (1+target_rate/100)**(1/(252/self.rebalance_window))-1
        sorted_portfolio_returns = np.sort(pd.Series(self.value_history).pct_change().dropna())
        cdf = np.arange(1, len(sorted_portfolio_returns) + 1) / len(sorted_portfolio_returns)
        cdf_value_at_k = np.interp(window_target_rate, sorted_portfolio_returns, cdf)           # Interpolate to find CDF value at k

        area_below_k = cdf_value_at_k                                                           # Area under the CDF for x < k
        area_above_k = 1 - cdf_value_at_k                                                       # Area above the CDF for x > k
        omega = area_above_k/area_below_k
        return omega
    
    def maximum_drawdown(self):
        values = self.value_history
        drawdowns = []
        max_so_far = values[0]
        for i in range(len(values)):
            if values[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = values[i]
            else:
                drawdown = (max_so_far - values[i]) / values[i]
                drawdowns.append(drawdown)
        return max(drawdowns)
    
    def avg_yearly_return(self):
        portfolio_history = pd.DataFrame({"date": pd.to_datetime(self.trading_dates), "value": self.value_history})
        annual_return_rates = portfolio_history.groupby(portfolio_history['date'].dt.year)['value'].apply(self.period_return).reset_index()
        annual_return_rates.columns = ["year", "return_rate"]
        return annual_return_rates["return_rate"].mean()
    
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
            portfolio_stock = Asset(name=stock, value=0, weighting=0, price=self.close_price.iloc[0][stock], num_shares=0)
            self.portfolio.append(portfolio_stock)
        
        # Initially only cash is held in the portfolio
        cash_asset = Asset(name="cash", value=self.principal, weighting=1, price=1, num_shares=self.principal)
        self.portfolio.append(cash_asset)

        self.last_day_tx_cost = 0

        # Initial observation
        curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                      # Close price of each asset at t
        prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                    # Close price of each asset at t-1
        log_return = np.log(np.divide(curr_close_price, prev_close_price))                              # Natural log of return
        rsi = np.array([x for x in self.rsi.iloc[self.time]])                                           # RSI of each asset at time t
        rsi = rsi / 100
        holdings = [asset.get_weighting() for asset in self.portfolio]                                  # Share and cash weightings 
        curr_close_price = (curr_close_price - curr_close_price.mean())/(curr_close_price.std())        # Normalize closing price
        prev_close_price = (prev_close_price - prev_close_price.mean())/(prev_close_price.std())

        initial_input = np.concatenate((curr_close_price, prev_close_price, log_return, rsi, holdings, [np.log(self.portfolio_value / self.principal)]))

        return initial_input

    # def step(self, action):
    #     done = 0
    #     self.time += 1
    #     old_portfolio_value = self.portfolio_value
    #     old_portfolio = copy.deepcopy(self.portfolio)

    #     # print("Time step:", self.time)

    #     # Compute the new portfolio value after 1 rebalance window
    #     # Price and value of a particular stock change in time = t+1, price of cash is unchanged
    #     new_value = 0
    #     for i in range(len(self.portfolio)):
    #         if (self.portfolio[i].get_name() != "cash"):
    #             self.portfolio[i].set_price(self.close_price.iloc[self.time][self.assets[i]])
    #         self.portfolio[i].set_value(self.portfolio[i].get_price() * self.portfolio[i].get_num_shares())
    #         new_value += self.portfolio[i].get_value()
        
    #     # Adjust the weighting of each asset in the portfolio based on the new portfolio value
    #     # An empty action array means skipping the portfolio rebalance, not applicable in RL algorithms
    #     if (len(action) != 0):
    #         for i in range(len(self.portfolio)):
    #             weight_adjusted_stock_value = new_value * action[i]
    #             self.portfolio[i].set_weighting(action[i])
    #             self.portfolio[i].set_num_shares(weight_adjusted_stock_value / self.portfolio[i].get_price())
    #             self.portfolio[i].set_value(weight_adjusted_stock_value)
    #     else:
    #         for i in range(len(self.portfolio)):
    #             self.portfolio[i].set_weighting(self.portfolio[i].get_value() / new_value)
        
    #     # Compute the transaction fee based on the number of shares bought/sold
    #     total_tx_cost = 0
    #     for i in range(len(self.portfolio)-1):
    #         total_tx_cost += abs(self.portfolio[i].get_num_shares() - old_portfolio[i].get_num_shares()) * self.tx_fee

    #     self.portfolio_value = new_value
    #     self.value_history.append(new_value)
    #     # reward = np.log(self.portfolio_value / old_portfolio_value)
    #     reward = self.portfolio_value - old_portfolio_value - total_tx_cost

    #     # New states
    #     curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                    # Close price of each asset at t
    #     prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                  # Close price of each asset at t-1
    #     log_return = np.array(np.log(np.divide(curr_close_price, prev_close_price)))                  # Natural log of return
    #     rsi = np.array([x for x in self.rsi.iloc[self.time]])                                         # RSI of each asset at time t
    #     rsi = rsi / 100
    #     holdings = np.array([asset.get_weighting() for asset in self.portfolio])                      # Share and cash holdings 
    #     curr_close_price = (curr_close_price - curr_close_price.mean())/(curr_close_price.std())      # Normalize closing price
    #     prev_close_price = (prev_close_price - prev_close_price.mean())/(prev_close_price.std())

    #     new_state = np.concatenate((curr_close_price, prev_close_price, log_return, rsi, holdings, [np.log(self.portfolio_value / self.principal)]))

    #     if (self.time == len(self.close_price)-1):                                                    # Indicate the end of the episode 
    #         done = 1

    #     return new_state, reward, done

    def step(self, action):
        done = 0
        old_portfolio_value = self.portfolio_value
        old_portfolio = copy.deepcopy(self.portfolio)

        # print("Time step:", self.time)

        # Compute the new portfolio value after 1 rebalance window
        # Price and value of a particular stock change in time = t+1, price of cash is unchanged
        new_value = 0
        for i in range(len(self.portfolio)):
            if (self.portfolio[i].get_name() != "cash"):
                self.portfolio[i].set_price(self.close_price.iloc[self.time][self.assets[i]])
            self.portfolio[i].set_value(self.portfolio[i].get_price() * self.portfolio[i].get_num_shares())
            new_value += self.portfolio[i].get_value()

        # Decucting ;ast day transaction fee from the portfolio value
        new_value -= self.last_day_tx_cost
        
        # Adjust the weighting of each asset in the portfolio based on the new portfolio value
        # An empty action array means skipping the portfolio rebalance, not applicable in RL algorithms
        if (len(action) != 0):
            for i in range(len(self.portfolio)):
                weight_adjusted_stock_value = new_value * action[i]
                self.portfolio[i].set_weighting(action[i])
                self.portfolio[i].set_num_shares(weight_adjusted_stock_value / self.portfolio[i].get_price())
                self.portfolio[i].set_value(weight_adjusted_stock_value)
        else:
            for i in range(len(self.portfolio)):
                self.portfolio[i].set_weighting(self.portfolio[i].get_value() / new_value)
        
        # Compute the transaction fee based on the number of shares bought/sold
        total_tx_cost = 0
        for i in range(len(self.portfolio)-1):
            total_tx_cost += abs(self.portfolio[i].get_num_shares() - old_portfolio[i].get_num_shares()) * self.tx_fee

        self.portfolio_value = new_value
        self.value_history.append(new_value)
        # reward = np.log(self.portfolio_value / old_portfolio_value)
        reward = self.portfolio_value - old_portfolio_value - total_tx_cost

        self.last_day_tx_cost = total_tx_cost

        self.time += 1

        # New states
        curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                    # Close price of each asset at t
        prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                  # Close price of each asset at t-1
        log_return = np.array(np.log(np.divide(curr_close_price, prev_close_price)))                  # Natural log of return
        rsi = np.array([x for x in self.rsi.iloc[self.time]])                                         # RSI of each asset at time t
        rsi = rsi / 100
        holdings = np.array([asset.get_weighting() for asset in self.portfolio])                      # Share and cash holdings 
        curr_close_price = (curr_close_price - curr_close_price.mean())/(curr_close_price.std())      # Normalize closing price
        prev_close_price = (prev_close_price - prev_close_price.mean())/(prev_close_price.std())

        new_state = np.concatenate((curr_close_price, prev_close_price, log_return, rsi, holdings, [np.log(self.portfolio_value / self.principal)]))

        if (self.time == len(self.close_price)-1):                                                    # Indicate the end of the episode 
            done = 1

        return new_state, reward, done
           
