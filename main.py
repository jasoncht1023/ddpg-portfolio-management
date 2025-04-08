import pandas as pd
from ddpg.agent_v2 import Agent
import numpy as np
from env.trading_simulator_v2 import TradingSimulator
import os
from scipy.optimize import minimize
import utils

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
    # "BMY",
    # "MMM",
    # "GS",
    # "NKE",
    # "AXP",
    # "HD",
    # "PG",
    # "AMGN",
    # "HON",
    # "CRM",
    # "AAPL",
    # "INTC",
    # "TRV",
    # "BA",
    # "IBM",
    # "UNH",
    # "CAT",
    # "JNJ",
    # "VZ",
    # "CVX",
    # "JPM",
    # "V",
    # "CSCO",
    # "MCD",
    # "WBA",
    # "KO",
    # "MRK",
    # "WMT",
    # "MSFT",
    # "DIS"
    # "AMD",
    # "ADDYY",
]
rebalance_window = 1
tx_fee_per_share = 0.005
principal = 1000000
num_episode = 500

# Either Training mode or Evaluation mode should be run at a time
is_training_mode = True

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
    "basic_MPT": 1
}

# Evaluation metrics
return_history = {}
sharpe_ratio_history = {}
actor_loss_history = []
critic_loss_history = []

# Trading environment initialization
env = TradingSimulator(principal=principal, assets=assets, start_date="2009-01-01", end_date="2017-12-31", 
                       rebalance_window=rebalance_window, tx_fee_per_share=tx_fee_per_share)

# Default: alpha=0.000025, beta=0.00025, gamma=0.99, tau=0.001, batch_size=64
agent = Agent(alpha=0.0001, beta=0.0005, gamma=0.99, tau=0.03, 
              input_dims=[len(assets) * 5 + 2], batch_size=128, n_actions=len(assets)+1)

# Training algorithms:
if (is_training_mode == True):
    if (training_mode["ddpg"] == 1):
        agent.load_models("trained_model")
        np.random.seed(0)
        return_history["ddpg"] = []
        sharpe_ratio_history["ddpg"] = []

        print("--------------------DDPG Training--------------------")
        for i in range(1, num_episode+1):
            print(f"-----------------Episode {i}-----------------")
            observation = env.restart()
            done = 0
            total_return = 0
            total_actor_loss = 0
            total_critic_loss = 0
            learning_count = 0

            while not done:
                action = agent.choose_action(observation, is_training_mode)
                new_state, reward, done = env.step(action)
                # if (i % 10 == 0 or i == 1):
                #     print("observation:", observation)
                #     print("action:", action, "\n")
                agent.remember(observation, action, reward, new_state, done)
                actor_loss, critic_loss = agent.learn() 
                if (actor_loss != None):
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss
                    learning_count += 1       
                total_return += reward
                # print("reward:", reward)
                observation = new_state
            
            # Append the metrics after a training episode is ended
            return_history["ddpg"].append(total_return)
            sharpe_ratio = env.sharpe_ratio()
            sharpe_ratio_history["ddpg"].append(sharpe_ratio)
            actor_loss_history.append(total_actor_loss / learning_count)
            critic_loss_history.append(total_critic_loss / learning_count)

            # Save the model and plot training progress graphs every 5 episodes
            if (i % 5 == 0):
                agent.save_models("trained_model")                
                episode_axis = range(1, i+1)
                utils.plot_return_over_episodes(episode_axis, return_history["ddpg"], "ddpg")
                utils.plot_sharpe_ratio_over_episodes(episode_axis, sharpe_ratio_history["ddpg"], "ddpg")
                utils.plot_mean_actor_loss_over_episodes(episode_axis, actor_loss_history, "ddpg")
                utils.plot_mean_critic_loss_over_episodes(episode_axis, critic_loss_history, "ddpg")
            print(f"------Episode {i} Summary: Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------\n")

            if (i == 100):
                agent.save_models("trained_model_100") 
            elif (i == 200):
                agent.save_models("trained_model_200")    
            elif (i == 300):
                agent.save_models("trained_model_300")    
            elif (i == 500):      
                agent.save_models("trained_model_500") 
        print("DDPG training done")
# Testing algorithms:
else:
    if (testing_mode["ddpg"] == 1):
        agent.load_models("trained_model")
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

        utils.print_eval_results(env, total_return)

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

        utils.print_eval_results(env, total_return)

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

        utils.print_eval_results(env, total_return)

    if (testing_mode["basic_MPT"] == 1):
        return_history["basic_MPT"] = []
        print("--------------------Efficient Frontier Tangent Portfolio--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        risk_free_rates = pd.read_csv('env/30y-treasury-rate.csv')
        risk_free_rates['date'] = pd.to_datetime(risk_free_rates['date']).dt.year
        risk_free_rates.columns = ["year", "risk_free_rate"]

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
            action = []
            if env.time > 3 and env.time % 1 == 0:
                t = max(env.time - 5, 0)
                r = env.close_price[t : env.time].pct_change().dropna()
                exp_r = r.mean()
                cov = r.cov()
                d = pd.to_datetime(env.trading_dates[env.time])
                risk_free_rate = risk_free_rates[risk_free_rates["year"] == d.year]["risk_free_rate"].values[0]
                
                # Calculate tangent portfolio weights
                weights = calculate_tangent_portfolio(exp_r, cov, risk_free_rate)
                
                action = list(weights) + [0]
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["basic_MPT"].append(total_return)
        
        utils.print_eval_results(env, total_return)

if not os.path.isdir("evaluation"): 
    os.makedirs("evaluation")

# Plotting training / testing evaluation graphs depending on the mode
if (is_training_mode == True):
    episode_axis = range(1, num_episode+1) 
    utils.plot_return_over_episodes(episode_axis, return_history["ddpg"], "ddpg")
    utils.plot_sharpe_ratio_over_episodes(episode_axis, sharpe_ratio_history["ddpg"], "ddpg")
    utils.plot_mean_actor_loss_over_episodes(episode_axis, actor_loss_history, "ddpg")
    utils.plot_mean_critic_loss_over_episodes(episode_axis, critic_loss_history, "ddpg")
else:
    utils.plot_testing_return(env, testing_mode, return_history)