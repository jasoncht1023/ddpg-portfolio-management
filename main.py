import pandas as pd
from ddpg.agent_v2 import Agent
import numpy as np
from env.trading_simulator_v2 import TradingSimulator
import matplotlib.pyplot as plt 
import os
from scipy.optimize import minimize

# Configurations
# Portfolio settings
assets = [
    "APA",
    "LNC",
    "RCL",
    "FCX",
    # "GOLD",
    # "FDP",
    # "NEM",
    # "BMY"
    # "MMM",
    # "NVDA",
    # "GS",
    # "NKE",
    # "AXP",
    # "HD",
    # "PG",
    # "AMGN",
    # "HON"
]
rebalance_window = 1
tx_fee_per_share = 0.005
principal = 1000000
num_epoch = 300

# Either Training mode or Evaluation mode should be run at a time
is_training_mode = False

# Training settings, 1: mode will be trained; 0: mode will not be run
training_mode = {
    "ddpg": 1
}

# Testing settings, 1: mode will be evaluated; 0: mode will not be run
# RL models must have a trained model to be evaluated
testing_mode = {
    "ddpg": 1,
    "uniform_with_rebalance": 1,
    "uniform_without_rebalance": 1,
    "basic_MPT": 0
}

# Evaluation metrics
return_history = {}
sharpe_ratio_history = {}

# Trading environment initialization
env = TradingSimulator(principal=principal, assets=assets, start_date="1999-07-01", end_date="2005-07-31", 
                       rebalance_window=rebalance_window, tx_fee_per_share=tx_fee_per_share)

# Default alpha=0.000025, beta=0.00025, gamma=0.99, tau=0.001, batch_size=64
agent = Agent(alpha=0.0005, beta=0.0025, gamma=0.99, tau=0.09, 
              input_dims=[len(assets) * 5 + 2], batch_size=128, n_actions=len(assets)+1)

actor_loss_history = []
critic_loss_history = []

# Training algorithms:
if (is_training_mode == True):
    if (training_mode["ddpg"] == 1):
        agent.load_models()
        np.random.seed(0)

        return_history["ddpg"] = []
        sharpe_ratio_history["ddpg"] = []
        print("--------------------DDPG Training--------------------")
        for i in range(1, num_epoch+1):
            print(f"-----------------Episode {i}-----------------")
            observation = env.restart()
            done = 0
            total_return = 0
            total_actor_loss = 0
            total_critic_loss = 0
            while not done:
                action = agent.choose_action(observation, is_training_mode)
                new_state, reward, done = env.step(action)
                # if (i % 10 == 0 or i == 1):
                #     # print("observation:", observation)
                #     # print("action:", action, "\n")
                agent.remember(observation, action, reward, new_state, done)
                actor_loss, critic_loss = agent.learn() 
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss       
                total_return += reward
                # print("reward:", reward)
                observation = new_state
            return_history["ddpg"].append(total_return)
            sharpe_ratio = env.sharpe_ratio()
            sharpe_ratio_history["ddpg"].append(sharpe_ratio)
            actor_loss_history.append(total_actor_loss)
            critic_loss_history.append(total_critic_loss)

            if i % 5 == 0:
                agent.save_models()
                xAxis = range(1, i+1)

                plt.title("Total return over epoch")
                plt.xlabel('Epoch') 
                plt.ylabel('Total return')
                plt.plot(xAxis, return_history["ddpg"], label="ddpg")
                plt.legend()
                plt.savefig("evaluation/training_total_return.png", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.title("Sharpe Ratio over epoch")
                plt.xlabel('Epoch') 
                plt.ylabel('Sharpe Ratio')   
                plt.plot(xAxis, sharpe_ratio_history["ddpg"], label="ddpg")
                plt.legend()
                plt.savefig("evaluation/training_sharpe_ratio.png", dpi=300, bbox_inches="tight")
                plt.clf()

                plt.title("Actor Loss")
                plt.xlabel('Progress') 
                plt.ylabel('Actor Loss')
                plt.plot(xAxis, actor_loss_history)
                plt.savefig("evaluation/actor_loss.png", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.title("Critic Loss")
                plt.xlabel('Progress') 
                plt.ylabel('Critic Loss')   
                plt.plot(xAxis, critic_loss_history)
                plt.savefig("evaluation/critic_loss.png", dpi=300, bbox_inches="tight")
                plt.clf()
            print(f"------Episode {i} Summary: Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------\n")

        print(f"DDPG average performance: Total Return {np.mean(return_history['ddpg'])}; Sharpe Ratio {np.mean(sharpe_ratio_history['ddpg'])}")
# Testing algorithms:
else:
    if (testing_mode["ddpg"] == 1):
        agent.load_models()
        np.random.seed(0)
        return_history["ddpg"] = []
        print("--------------------DDPG--------------------")
        observation = env.restart()
        done = 0
        total_return = 0
        while not done:
            action = agent.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg"].append(total_return)
        sharpe_ratio = env.sharpe_ratio()
        portfolio_value = env.total_portfolio_value()
        print(f"------Portfolio Value {portfolio_value:.2f}; Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------\n")

    if (testing_mode["uniform_with_rebalance"] == 1):
        return_history["uniform_with_rebalance"] = []
        print("--------------------Uniform Weighting with Rebalancing--------------------")
        observation = env.restart()
        done = 0
        total_return = 0
        while not done:
            action = [1/(len(assets))] * (len(assets)) + [0]
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["uniform_with_rebalance"].append(total_return)
        sharpe_ratio = env.sharpe_ratio()
        portfolio_value = env.total_portfolio_value()
        print(f"------Portfolio Value {portfolio_value:.2f}; Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------\n")

    if (testing_mode["uniform_without_rebalance"] == 1):
        return_history["uniform_without_rebalance"] = []
        print("--------------------Uniform Weighting without Rebalancing--------------------")
        observation = env.restart()
        done = 0
        total_return = 0
        action = [1/(len(assets))] * (len(assets)) + [0]
        new_state, reward, done = env.step(action)
        total_return += reward
        return_history["uniform_without_rebalance"].append(total_return)
        while not done:
            action = []
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["uniform_without_rebalance"].append(total_return)
        sharpe_ratio = env.sharpe_ratio()
        portfolio_value = env.total_portfolio_value()
        print(f"------Portfolio Value {portfolio_value:.2f}; Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------\n")

    if (testing_mode["basic_MPT"] == 1):
        return_history["basic_MPT"] = []
        print("--------------------Efficient Frontier Tangent Portfolio--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        risk_free_rates.columns = ["date", "risk_free_rate"]
        rebalance_dates_df = pd.DataFrame(env.rebalance_dates, columns=["date"])
        risk_free_rates = pd.merge(rebalance_dates_df, risk_free_rates, on="date", how="left")
        risk_free_rates["risk_free_rate"] = risk_free_rates["risk_free_rate"] * env.rebalance_window / 365
        risk_free_rates["risk_free_rate"] = risk_free_rates["risk_free_rate"].interpolate().bfill().ffill()

        def calculate_tangent_portfolio(exp_r, cov, risk_free_rate):
            num_assets = len(exp_r)
            
            def portfolio_performance(weights, exp_r, cov, risk_free_rate):
                returns = np.dot(weights, exp_r)
                std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
                sharpe_ratio = (returns - risk_free_rate) / std
                return -sharpe_ratio  # Negative because we minimize

            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets,]

            result = minimize(portfolio_performance, initial_guess, args=(exp_r, cov, risk_free_rate),
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            return result.x

        while not done:
            # t = max(env.time - 4, 0) # MPT should use more days to calulate the covariance matrix (but fair test with DDPG the range should depends on the window of the matrix for DDPG)
            t = env.time
            r = env.close_price[t * rebalance_window : (t + 1) * rebalance_window].pct_change().dropna()
            exp_r = r.mean()
            cov = r.cov()
            d = env.rebalance_dates[t]
            risk_free_rate = risk_free_rates[risk_free_rates["date"] == d]["risk_free_rate"].values[0]
            
            # Calculate tangent portfolio weights
            weights = calculate_tangent_portfolio(exp_r, cov, risk_free_rate)
            
            action = list(weights) + [0]
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["basic_MPT"].append(total_return)
        sharpe_ratio = env.sharpe_ratio()
        print(f"------Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------\n")

if not os.path.isdir("evaluation"): 
    os.makedirs("evaluation")

# Plotting training / testing evaluation graphs depending on the mode
if (is_training_mode == True):
    xAxis = range(1, num_epoch+1) 

    plt.title("Total return over epoch")
    plt.xlabel('Epoch') 
    plt.ylabel('Total return')

    for mode in training_mode:
        if (training_mode[mode] == 1):
            plt.plot(xAxis, return_history[mode], label=mode)

    plt.legend()
    plt.savefig("evaluation/training_total_return.png", dpi=300, bbox_inches="tight")
    plt.clf()

    plt.title("Sharpe Ratio over epoch")
    plt.xlabel('Epoch') 
    plt.ylabel('Sharpe Ratio')   

    for mode in training_mode:
        if (training_mode[mode] == 1):
            plt.plot(xAxis, sharpe_ratio_history[mode], label=mode)

    plt.legend()
    plt.savefig("evaluation/training_sharpe_ratio.png", dpi=300, bbox_inches="tight")
else:
    xAxis = range(1, len(return_history[list(return_history.keys())[0]])+1)               # Get the number of times of portfolio rebalance

    plt.title("Cumulative return over time")
    plt.xlabel('Time') 
    plt.ylabel('Cumulative return')

    for mode in testing_mode:
        if (testing_mode[mode] == 1):
            plt.plot(xAxis, return_history[mode], label=mode)

    plt.legend()
    plt.savefig("evaluation/test_cumulative_return.png", dpi=300, bbox_inches="tight")
    plt.clf()