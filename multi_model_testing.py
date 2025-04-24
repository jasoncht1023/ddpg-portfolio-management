import pandas as pd
from ddpg.agent import Agent
import numpy as np
from env.trading_simulator import TradingSimulator
import os
from scipy.optimize import minimize
import utils

# Configurations
# Portfolio settings
assets = [
    # "APA",
    # "LNC",
    # "RCL",
    # "FCX",
    # "GOLD",
    # "FDP",
    # "NEM",
    # "MMM",
    # "GS",
    # "AXP",
    # "HD",
    # "PG",
    # "AMGN",
    # "HON",
    # "CRM",
    # "AAPL",
    # "TRV",
    # "IBM",
    # "UNH",
    # "CAT",
    # "JNJ",
    # "JPM",
    # "V",
    # "MCD",
    # "WBA",
    # "KO",
    # "WMT",
    # "MSFT",
    # "DIS",
    # "ADDYY",
    # "AMD",
    # "BA",
    # "SBUX",
    # "TLT",
    # "GLD",
    # "NKE", 
    # "CVX"
    "CSCO", 
    "BMY", 
    "BAC", 
    "TGT", 
    "VZ", 
    "GE", 
    "CL", 
    "CVX", 
    "DUK", 
    "PLD", 
    "LIN", 
    "INTC", 
    "MRK", 
    "WFC",
]
rebalance_window = 1
tx_fee_per_share = 0.005
principal = 1000000
num_episode = 500

# Either Training mode or Evaluation mode should be run at a time
is_training_mode = False

is_saved_models_zipped = True

# Training settings, 1: mode will be trained; 0: mode will not be run
training_mode = {
    "ddpg": 1
}

# Testing settings, 1: mode will be evaluated; 0: mode will not be run
# RL models must have a trained model to be evaluated
testing_mode = {
    "uniform_without_rebalance": 0,
    "uniform_with_rebalance": 0,
    "all_in_last_day_best_return": 0,
    "follow_last_day_best_return": 0,
    "mpt": 0,
    "ddpg_fc": 0,
    "ddpg_lstm": 1,
    "ddpg_lstm_longer": 0,
    "ddpg_lstm_shorter": 0,
    "ddpg_lstm_0": 0,
    "ddpg_lstm_100": 0,
    "ddpg_lstm_200": 0,
    "ddpg_lstm_300": 0,
    "ddpg_lstm_500": 0,
    "ddpg_amplifier": 0,
    "god": 0,
}

# Evaluation metrics
return_history = {}
yearly_return_rate_history = {}
monthly_return_rate_history = {}
sharpe_ratio_history = {}
actor_loss_history = []
critic_loss_history = []

# Trading environment initialization (Trained: 2009-2021)
env = TradingSimulator(principal=principal, assets=assets, start_date="2022-01-01", end_date="2024-12-31", 
                        rebalance_window=rebalance_window, tx_fee_per_share=tx_fee_per_share)

agent1 = Agent(alpha=0.0005, beta=0.0025, gamma=0.99, tau=0.09, input_dims=[len(assets) * 8 + 1], 
              batch_size=128, n_actions=len(assets) + 1, model=1)

agent2 = Agent(alpha=0.0005, beta=0.0025, gamma=0.99, tau=0.09, input_dims=[len(assets) * 8 + 1], 
              batch_size=128, n_actions=len(assets) + 1, model=2)

agent3 = Agent(alpha=0.0005, beta=0.0025, gamma=0.99, tau=0.09, input_dims=[len(assets) * 8 + 1], 
              batch_size=128, n_actions=len(assets), model=3)

# Training algorithms:
if (is_training_mode == True):
    pass
# Testing algorithms:
else:
    if (testing_mode["ddpg_fc"] == 1):
        agent1.load_models("test_48_500", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_fc"] = []

        print("--------------------DDPG FC--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent1.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_fc"].append(total_return)

        yearly_return_rate_history["ddpg_fc"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_fc"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)
    
    if (testing_mode["ddpg_lstm"] == 1):
        agent2.load_models("test_420", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_lstm"] = []

        print("--------------------DDPG LSTM--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent2.choose_action(observation, is_training_mode)
            print(action)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_lstm"].append(total_return)

        yearly_return_rate_history["ddpg_lstm"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_lstm"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)
    
    # if (testing_mode["ddpg_lstm_longer"] == 1):
    #     agent2.load_models("test_48_500", is_saved_models_zipped)
    #     np.random.seed(0)
    #     return_history["ddpg_lstm_longer"] = []

    #     print("--------------------DDPG LSTM--------------------")
    #     observation = env.restart()
    #     done = 0
    #     total_return = 0

    #     while not done:
    #         action = agent2.choose_action(observation, is_training_mode)
    #         # print(action, "\n")
    #         new_state, reward, done = env.step(action)
    #         # print(new_state, "\n")
    #         total_return += reward
    #         observation = new_state
    #         return_history["ddpg_lstm_longer"].append(total_return)

    #     yearly_return_rate_history["ddpg_lstm_longer"], _ = env.yearly_return_history()
    #     monthly_return_rate_history["ddpg_lstm_longer"], _ = env.monthly_return_history()
    #     utils.print_eval_results(env, total_return)

    # if (testing_mode["ddpg_lstm_shorter"] == 1):
    #     agent2.load_models("test_55_500", is_saved_models_zipped)
    #     np.random.seed(0)
    #     return_history["ddpg_lstm_shorter"] = []

    #     print("--------------------DDPG LSTM--------------------")
    #     observation = env.restart()
    #     done = 0
    #     total_return = 0

    #     while not done:
    #         action = agent2.choose_action(observation, is_training_mode)
    #         # print(action, "\n")
    #         new_state, reward, done = env.step(action)
    #         # print(new_state, "\n")
    #         total_return += reward
    #         observation = new_state
    #         return_history["ddpg_lstm_shorter"].append(total_return)

    #     yearly_return_rate_history["ddpg_lstm_shorter"], _ = env.yearly_return_history()
    #     monthly_return_rate_history["ddpg_lstm_shorter"], _ = env.monthly_return_history()
    #     utils.print_eval_results(env, total_return)

    if (testing_mode["ddpg_lstm_0"] == 1):
        agent2.load_models("test_48_0", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_lstm_0"] = []

        print("--------------------DDPG 0--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent2.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_lstm_0"].append(total_return)

        yearly_return_rate_history["ddpg_lstm_0"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_lstm_0"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["ddpg_lstm_100"] == 1):
        agent2.load_models("test_48_100", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_lstm_100"] = []

        print("--------------------DDPG 100--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent2.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_lstm_100"].append(total_return)

        yearly_return_rate_history["ddpg_lstm_100"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_lstm_100"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["ddpg_lstm_200"] == 1):
        agent2.load_models("test_48_200", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_lstm_200"] = []

        print("--------------------DDPG 200--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent2.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_lstm_200"].append(total_return)

        yearly_return_rate_history["ddpg_lstm_200"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_lstm_200"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["ddpg_lstm_300"] == 1):
        agent2.load_models("test_48_300", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_lstm_300"] = []

        print("--------------------DDPG 300--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent2.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_lstm_300"].append(total_return)

        yearly_return_rate_history["ddpg_lstm_300"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_lstm_300"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["ddpg_lstm_500"] == 1):
        agent2.load_models("test_48_500", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_lstm_500"] = []

        print("--------------------DDPG 500--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent2.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_lstm_500"].append(total_return)

        yearly_return_rate_history["ddpg_lstm_500"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_lstm_500"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)
    
    if (testing_mode["ddpg_amplifier"] == 1):
        agent3.load_models("test_56", is_saved_models_zipped)
        np.random.seed(0)
        return_history["ddpg_amplifier"] = []

        print("--------------------DDPG Amplifier--------------------")
        observation = env.restart()
        done = 0
        total_return = 0

        while not done:
            action = agent3.choose_action(observation, is_training_mode)
            # print(action, "\n")
            new_state, reward, done = env.step(action)
            # print(new_state, "\n")
            total_return += reward
            observation = new_state
            return_history["ddpg_amplifier"].append(total_return)

        yearly_return_rate_history["ddpg_amplifier"], _ = env.yearly_return_history()
        monthly_return_rate_history["ddpg_amplifier"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["god"] == 1):
        return_history["god"] = []

        print("--------------------God--------------------")
        observation = env.restart()
        done = 0
        total_return = 0
        n = len(assets)

        while not done:
            action = [0] * (n+1)
            if env.time < len(env.close_price) - 2:
                action_close_price = np.array([x for x in env.close_price.iloc[env.time]])
                forward_close_price = np.array([x for x in env.close_price.iloc[env.time+1]])
                logr = np.log(np.divide(forward_close_price, action_close_price))
                if np.max(logr) >= 0:
                    action[np.argmax(logr)] = 1
                else:
                    action[-1] = 1
            else:
                action[-1] = 1
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["god"].append(total_return)

        yearly_return_rate_history["god"], _ = env.yearly_return_history()
        monthly_return_rate_history["god"], _ = env.monthly_return_history()
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

        yearly_return_rate_history["uniform_without_rebalance"], _ = env.yearly_return_history()
        monthly_return_rate_history["uniform_without_rebalance"], _ = env.monthly_return_history()
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

        yearly_return_rate_history["uniform_with_rebalance"], _ = env.yearly_return_history()
        monthly_return_rate_history["uniform_with_rebalance"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["follow_last_day_best_return"] == 1):
        return_history["follow_last_day_best_return"] = []

        print("--------------------follow last day best return--------------------")
        observation = env.restart()
        done = 0
        total_return = 0
        n = len(assets)

        prev_action = [1/(len(assets)+1)] * (len(assets)+1)

        while not done:
            action = prev_action.copy()

            curr_close_price = np.append(np.array(env.close_price.iloc[env.time]), 1)
            prev_close_price = np.append(np.array(env.close_price.iloc[env.time - 1]), 1)

            logr = np.log(np.divide(prev_close_price, curr_close_price))        # logarithmic returns

            # Increase the asset with the maximum logarithmic return
            max_index = np.argmax(logr)
            action[max_index] += 0.1

            total_decrease_needed = 0.1
            sorted_indices = np.argsort(logr)

            for idx in sorted_indices:
                amount_available = action[idx]
                subtraction = min(amount_available, total_decrease_needed)
                action[idx] -= subtraction
                total_decrease_needed -= subtraction

                if (total_decrease_needed == 0):
                    break

            prev_action = action
            
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["follow_last_day_best_return"].append(total_return)

        yearly_return_rate_history["follow_last_day_best_return"], _ = env.yearly_return_history()
        monthly_return_rate_history["follow_last_day_best_return"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

    if (testing_mode["all_in_last_day_best_return"] == 1):
        return_history["all_in_last_day_best_return"] = []

        print("--------------------All-in last day best return--------------------")
        observation = env.restart()
        done = 0
        total_return = 0
        n = len(assets)

        while not done:
            action = [0] * (n+1)
            curr_close_price = np.array([x for x in env.close_price.iloc[env.time]])
            prev_close_price = np.array([x for x in env.close_price.iloc[env.time-1]])
            logr = np.log(np.divide(prev_close_price, curr_close_price))
            if np.max(logr) >= 0:
                action[np.argmax(logr)] = 1
            else:
                action[-1] = 1
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["all_in_last_day_best_return"].append(total_return)

        yearly_return_rate_history["all_in_last_day_best_return"], _ = env.yearly_return_history()
        monthly_return_rate_history["all_in_last_day_best_return"], _ = env.monthly_return_history()
        utils.print_eval_results(env, total_return)

if (testing_mode["mpt"] == 1):
        return_history["mpt"] = []
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
            window = 30
            if env.time > 3:
                t = max(env.time - window, 0)
                r = env.close_price[t : env.time].pct_change().dropna()
                exp_r = r.mean()
                cov = r.cov()
                d = pd.to_datetime(env.trading_dates[env.time])
                risk_free_rate = (1+risk_free_rates[risk_free_rates["year"] == d.year]["risk_free_rate"].values[0]/100)**(1/252 * min(env.time, window)) - 1
                
                # Calculate tangent portfolio weights
                weights = calculate_tangent_portfolio(exp_r, cov, risk_free_rate)
                
                action = list(weights) + [0]
            new_state, reward, done = env.step(action)
            total_return += reward
            return_history["mpt"].append(total_return)

        yearly_return_rate_history["mpt"], _ = env.yearly_return_history()
        monthly_return_rate_history["mpt"], _ = env.monthly_return_history()        
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
    utils.plot_testing_return(env, testing_mode, return_history, yearly_return_rate_history, monthly_return_rate_history)