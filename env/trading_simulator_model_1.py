import math
import yfinance as yf
import pandas as pd
import numpy as np
import copy
from datetime import datetime, timedelta
from env.asset import Asset
from scipy.optimize import minimize

class TradingSimulator:
    def __init__(self, principal, assets, start_date, end_date, rebalance_window, tx_fee_per_share):
        compute_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
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
            returns_series = adjusted_close.pct_change(fill_method=None)
            # Append the Series to the list
            returns_list.append(returns_series.rename(stock))  # Rename for clarity

        # Concatenate all return Series into a single DataFrame
        returns = pd.concat(returns_list, axis=1)

        returns.reset_index(inplace=True)
        returns = returns.set_index('Date')

        dates = returns.index
        adj_close = self.data.xs("Close", level=1, axis=1)
        adj_close = adj_close.reindex(columns=returns.columns)
        volume = self.data.xs("Volume", level=1, axis=1)
        volume = volume.reindex(columns=returns.columns)

        columns = pd.MultiIndex.from_product([assets, ['Close', 'Returns', "RSI", "MA20", "Std", "DevToMidBand", "Volume", "EMA_12", "EMA_26", "MACD", "signal", "OBV"]])
        df = pd.DataFrame(index=dates, columns=columns)
        df.columns = columns
        for stock in assets:
            df[(stock, "Close")] = adj_close[stock]
            df[(stock, "Returns")] = returns[stock]
            df[(stock, "Volume")] = volume[stock]
        df = df.reset_index()

        for stock in assets:
            df[(stock, "RSI")] = df[(stock, "Returns")].rolling(2).apply(self.__RSI)
            max_vol_scale = math.floor(math.log10(df[(stock, "Volume")].max()))
            df[(stock, "Volume")] = df[(stock, "Volume")].apply(lambda x: x / (10 ** max_vol_scale))
            df[(stock, "OBV")] = (np.sign(df[(stock, "Returns")]) * df[(stock, "Volume")]).cumsum()
            df[(stock, "OBV")] = self.min_max_scaling(df[(stock, "OBV")])
            df[(stock, "MA20")] = df[(stock, "Close")].rolling(window=20).mean()
            df[(stock, "Std")] = df[(stock, "Close")].rolling(window=20).std()
            df[(stock, "DevToMidBand")] = (df[(stock, "Close")] - df[(stock, "MA20")]) / df[(stock, "Std")]
            df.loc[0, (stock, "EMA_12")] = df.loc[0, (stock, "Close")]
            df.loc[0, (stock, "EMA_26")] = df.loc[0, (stock, "Close")]
            for i in range(1, len(df)):
                df.loc[i, (stock, "EMA_12")] = self.__EMA(12, df.loc[i, (stock, "Close")], df.loc[i-1, (stock, "EMA_12")])
                df.loc[i, (stock, "EMA_26")] = self.__EMA(26, df.loc[i, (stock, "Close")], df.loc[i-1, (stock, "EMA_26")])

            df[(stock, "MACD")] = df[(stock, "EMA_12")] - df[(stock, "EMA_26")]
            df.loc[0, (stock, "signal")] = df.loc[0, (stock, "MACD")]
            for i in range(1, len(df)):
                df.loc[i, (stock, "signal")] = self.__EMA(9, df.loc[i, (stock, "MACD")], df.loc[i-1, (stock, "signal")])

        close_data = df[1:len(df)]                                                                          # Drop the first row without the RSI value
        close_data = close_data.reset_index(drop=True)
        to_drop = ["Returns", "MA20", "Std", "Volume", "EMA_12", "EMA_26"]                                  # Drop the columns that are not needed
        close_data = close_data.drop(columns=[(stock, label) for stock in assets for label in to_drop])     # Drop unused columns
        start_index = close_data[close_data['Date'] >= start_date].index[0]                                 # Index of the first trading date in range
        close_data = close_data[start_index-1:].reset_index(drop=True)                                      # Take an extra day before the first in-range trading date
        def numpy_rolling_cov(returns, window_size):
            n_stocks = returns.shape[1]
            n_days = returns.shape[0]
            rolling_cov = np.zeros((n_days - window_size + 1, n_stocks, n_stocks))
            
            for t in range(n_days - window_size + 1):
                window = returns[t:t + window_size]
                cov = np.cov(window.T, ddof=0)  # Covariance
                rolling_cov[t] = cov
            
            return rolling_cov
        
        def read_treasury_rates(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Convert the 'date' column to datetime
            df['date'] = pd.to_datetime(df['date'])
            # Extract the year from the 'date' column
            df['year'] = df['date'].dt.year
            # Create a dictionary of year and value pairs
            treasury_rate = dict(zip(df['year'], df[' value']))
            return treasury_rate
        
        self.trading_dates = close_data["Date"].dt.date.astype(str).tolist()[1:]
        # self.rebalance_dates = [self.trading_dates[i] for i in range(len(self.trading_dates)) if (i+1) % rebalance_window == 0]
        mpt_window = 20

        rolling_cov = numpy_rolling_cov(returns.to_numpy(), mpt_window)[-len(self.trading_dates)-1:]
        rolling_exp_returns = returns.rolling(mpt_window).mean()[-len(self.trading_dates)-1:]

        file_path = './env/30y-treasury-rate.csv'
        treasury_rate = read_treasury_rates(file_path)
        self.tangent_portfolios = []
        for i in range(len(rolling_cov)):
            cov = rolling_cov[i]
            exp_r = rolling_exp_returns.iloc[i].values
            year = rolling_exp_returns.iloc[i].name.year
            rate = (1 + treasury_rate[year]/100.0)**(1/252 * mpt_window) - 1
            self.tangent_portfolios.append(self.calculate_tangent_portfolio(exp_r, cov, rate))

        # Collect all the Adjusted Close price data
        adj_close_data = close_data.loc[:, close_data.columns.get_level_values(1) == "Close"]
        adj_close_data.columns = adj_close_data.columns.droplevel(1)
        self.close_price = adj_close_data

        rsi_data = close_data.loc[:, close_data.columns.get_level_values(1) == "RSI"]
        rsi_data.columns = rsi_data.columns.droplevel(1)
        self.rsi = rsi_data

        dev_data = close_data.loc[:, close_data.columns.get_level_values(1) == "DevToMidBand"]
        dev_data.columns = dev_data.columns.droplevel(1)
        self.dev = dev_data

        macd_data = close_data.loc[:, close_data.columns.get_level_values(1) == "MACD"]
        macd_data.columns = macd_data.columns.droplevel(1)
        self.macd = macd_data

        signal_data = close_data.loc[:, close_data.columns.get_level_values(1) == "signal"]
        signal_data.columns = signal_data.columns.droplevel(1)
        self.signal = signal_data

        obv_data = close_data.loc[:, close_data.columns.get_level_values(1) == "OBV"]
        obv_data.columns = obv_data.columns.droplevel(1)
        self.obv = obv_data

        # Stays constant throughout epochs
        self.principal = principal
        self.assets = assets
        self.rebalance_window = rebalance_window
        self.tx_fee = tx_fee_per_share
        self.risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        self.risk_free_rates['date'] = pd.to_datetime(self.risk_free_rates['date']).dt.year
        self.risk_free_rates.columns = ["year", "risk_free_rate"]

    def min_max_scaling(self, data):
        min_val = data.min()
        max_val = data.max()
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data
    
    def __EMA(self, w, price, last):
        a = 2/(1+w)
        return a*price + (1-a)*last

    def __RSI(self, returns):
        if (len(returns[returns > 0]) == 0):
            return 0
        if (len(returns[returns < 0]) == 0):
            return 100
        avg_gain = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        return 100 * (1 - 1/(1+avg_gain/avg_loss))
    
    def calculate_tangent_portfolio(self, expected_returns, covariance_matrix, risk_free_rate):

        n = len(expected_returns)
        
        def objective_function(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            sharpe_ratio = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
            return -sharpe_ratio
        
        # Define the constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights must be 1
            {'type': 'ineq', 'fun': lambda x: x}  # No short-selling (all weights must be non-negative)
        ]
        
        # Define the initial guess
        initial_guess = np.ones(n) / n
        
        # Solve the optimization problem
        result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP')
        
        # Return the optimal weights
        return result.x
    
    # Compute the rate of return of the whole period
    def period_return(self, periood_return_history):
        start_value = periood_return_history.iloc[0] 
        end_value = periood_return_history.iloc[-1] 
        return (end_value - start_value) / start_value * 100 
    
    def sharpe_ratio(self):     
        def annual_return(yearly_return_history):
            start_value = yearly_return_history.iloc[0] 
            end_value = yearly_return_history.iloc[-1] 
            return (end_value - start_value) / start_value
        
        # Load the annual risk free rates
        risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        risk_free_rates['date'] = pd.to_datetime(risk_free_rates['date']).dt.year
        risk_free_rates.columns = ["year", "risk_free_rate"]

        # Compute the return rates
        portfolio_history = pd.DataFrame({"date": pd.to_datetime(self.trading_dates), "value": self.value_history})
        annual_return_rates = portfolio_history.groupby(portfolio_history['date'].dt.year)['value'].apply(annual_return).reset_index()
        annual_return_rates.columns = ["year", "return_rate"]

        # Combine the risk free rates and return rates
        df = annual_return_rates.set_index('year').join(risk_free_rates.set_index('year'))

        df["excess_return_rate"] = df["return_rate"] - df["risk_free_rate"]/100

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
        drawdowns = [0]
        max_so_far = values[0]
        for i in range(len(values)):
            if values[i] > max_so_far:
                max_so_far = values[i]
            else:
                drawdown = (max_so_far - values[i]) / max_so_far
                drawdowns.append(drawdown)
        return max(drawdowns)
    
    def avg_yearly_return(self):
        portfolio_history = pd.DataFrame({"date": pd.to_datetime(self.trading_dates), "value": self.value_history})
        annual_return_rates = portfolio_history.groupby(portfolio_history['date'].dt.year)['value'].apply(self.period_return).reset_index()
        annual_return_rates.columns = ["year", "return_rate"]
        return annual_return_rates["return_rate"].mean()
    
    def avg_monthly_return(self):
        portfolio_history = pd.DataFrame({"date": pd.to_datetime(self.trading_dates), "value": self.value_history})
        monthly_return_rates = portfolio_history.groupby(portfolio_history['date'].dt.to_period("M"))['value'].apply(self.period_return).reset_index()
        monthly_return_rates.columns = ["month", "return_rate"]
        return monthly_return_rates["return_rate"].mean()
    
    def total_portfolio_value(self):
        return self.portfolio_value
    
    def trading_date_range(self):
        return self.trading_dates[1:]
    
    def get_tangent_weights(self, lag):
        t = max(self.time - lag, 0)
        r = self.close_price[t : self.time].pct_change().dropna()
        exp_r = r.mean()
        cov = r.cov()
        d = pd.to_datetime(self.trading_dates[self.time])
        risk_free_rate = self.risk_free_rates[self.risk_free_rates["year"] == d.year]["risk_free_rate"].values[0]
        weights = self.calculate_tangent_portfolio(exp_r, cov, risk_free_rate)
        return list(weights)

    def restart(self):
        # Reset the initial portfolio
        self.portfolio = []
        self.portfolio_value = self.principal
        self.time = 1

        # Portfolio value history for return computation used in evaluation
        self.value_history = [self.principal]
        self.tx_cost_history = [0]

        # Initializing the stock part of the portfolio
        for stock in self.assets:
            portfolio_stock = Asset(name=stock, value=0, weighting=0, price=self.close_price.iloc[0][stock], num_shares=0)
            self.portfolio.append(portfolio_stock)
        
        # Initially only cash is held in the portfolio
        cash_asset = Asset(name="cash", value=self.principal, weighting=1, price=1, num_shares=self.principal)
        self.portfolio.append(cash_asset)

        # Initial observation
        curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                      # Close price of each asset at t
        prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                    # Close price of each asset at t-1
        log_return = np.log(np.divide(curr_close_price, prev_close_price))                              # Natural log of return
        prev_rsi = np.array([x for x in self.rsi.iloc[self.time-1]]) / 100.0                              # RSI of each asset at time t-1
        curr_rsi = np.array([x for x in self.rsi.iloc[self.time]]) / 100.0                                           # RSI of each asset at time t
        dev = np.array([x for x in self.dev.iloc[self.time]])                                         # No. of Standard Deviation between price and the mid BollingerBand (MA20) of each asset at time t
        obv = np.array([x for x in self.obv.iloc[self.time]])
        macd = np.array([x for x in self.macd.iloc[self.time]])
        macd = (macd - macd.mean())/(macd.std())
        signal = np.array([x for x in self.signal.iloc[self.time]])
        signal = (signal - signal.mean())/(signal.std())

        holdings = [asset.get_weighting() for asset in self.portfolio]                                  # Share and cash weightings 
        curr_close_price = (curr_close_price - curr_close_price.mean())/(curr_close_price.std())        # Normalize closing price
        prev_close_price = (prev_close_price - prev_close_price.mean())/(prev_close_price.std())
        tangent_portfolio = self.tangent_portfolios[self.time]                                          # Tangent portfolio weights

        initial_input = np.concatenate((log_return, dev, obv, macd, signal, curr_rsi, prev_rsi, holdings))

        return initial_input

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
        new_value -= self.tx_cost_history[-1]
        
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
        self.tx_cost_history.append(total_tx_cost)

        self.portfolio_value = new_value
        self.value_history.append(new_value)
        # reward = np.log(self.portfolio_value / old_portfolio_value)
        reward = self.portfolio_value - old_portfolio_value - total_tx_cost

        self.time += 1

        # New states
        curr_close_price = np.array([x for x in self.close_price.iloc[self.time]])                    # Close price of each asset at t
        prev_close_price = np.array([x for x in self.close_price.iloc[self.time-1]])                  # Close price of each asset at t-1
        log_return = np.array(np.log(np.divide(curr_close_price, prev_close_price)))                  # Natural log of return
        prev_rsi = np.array([x for x in self.rsi.iloc[self.time-1]]) / 100.0                              # RSI of each asset at time t-1
        curr_rsi = np.array([x for x in self.rsi.iloc[self.time]]) / 100.0
        dev = np.array([x for x in self.dev.iloc[self.time]])                                         # No. of Standard Deviation between price and the mid BollingerBand (MA20) of each asset at time t
        obv = np.array([x for x in self.obv.iloc[self.time]])
        macd = np.array([x for x in self.macd.iloc[self.time]])
        macd = (macd - macd.mean())/(macd.std())
        signal = np.array([x for x in self.signal.iloc[self.time]])
        signal = (signal - signal.mean())/(signal.std())
        holdings = np.array([asset.get_weighting() for asset in self.portfolio])                      # Share and cash holdings 

        new_state = np.concatenate((log_return, dev, obv, macd, signal, curr_rsi, prev_rsi, holdings))  # Concatenate the new state variables

        if (self.time == len(self.close_price)-1):                                                    # Indicate the end of the episode 
            done = 1

        return new_state, reward, done