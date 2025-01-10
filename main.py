from ddpg.agent import Agent
import numpy as np
from env.trading_simulator import TradingSimulator
import matplotlib.pyplot as plt 
import os

# Configurations
# Portfolio settings
assets = [
    "MMM",
    "NVDA",
    "GS",
    "NKE",
    "AXP",
    "HD",
    "PG",
    "AMGN",
    "HON"
]
rebalance_window = 10
tx_fee_per_share = 0.005
principal = 1000000
num_epoch = 10

# Evaluation settings, 1: mode will be evaluated; 0: mode will not be run
evaluation_mode = {
    "ddpg": 1,
    "uniform_with_rebalance": 1,
    "uniform_without_rebalance": 1
}

# Evaluation metrics
return_history = {}
sharpe_ratio_history = {}

# Trading environment initialization
env = TradingSimulator(principal=principal, assets=assets, start_date="2024-01-01", end_date="2024-12-31", 
                       rebalance_window=rebalance_window, tx_fee_per_share=tx_fee_per_share)

if (evaluation_mode["ddpg"] == 1):
    # Default alpha=0.000025, beta=0.00025, tau=0.001, batch_size=64
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4, len(assets), rebalance_window, len(assets)],
                tau=0.001, batch_size=8, n_actions=len(assets)+1)

    agent.load_models()
    np.random.seed(0)

    return_history["ddpg"] = []
    sharpe_ratio_history["ddpg"] = []
    print("--------------------DDPG--------------------")
    for i in range(num_epoch):
        print(f"-----------------Episode {i+1}-----------------")
        observation = env.restart()
        done = 0
        total_return = 0
        while not done:
            action = agent.choose_action(observation)
            print("action:", action)
            new_state, reward, done = env.step(action)
            print("reward:", reward)
            agent.remember(observation, action, reward, new_state, done)
            agent.learn()
            total_return += reward
            observation = new_state
        return_history["ddpg"].append(total_return)
        sharpe_ratio = env.sharpe_ratio()
        sharpe_ratio_history["ddpg"].append(sharpe_ratio)

        if i % 5 == 0:
            agent.save_models()
        print(f"------Episode {i+1} Summary: Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------")

if (evaluation_mode["uniform_with_rebalance"] == 1):
    print("--------------------Uniform Weighting with Rebalancing--------------------")
    observation = env.restart()
    done = 0
    total_return = 0
    while not done:
        action = [1/(len(assets)+1)] * (len(assets)+1)
        print("action:", action)
        new_state, reward, done = env.step(action)
        print("reward:", reward)
        total_return += reward
    sharpe_ratio = env.sharpe_ratio()

    return_history["uniform_with_rebalance"]= [total_return] * num_epoch
    sharpe_ratio_history["uniform_with_rebalance"] = [sharpe_ratio] * num_epoch

    print(f"------Summary: Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------")

if (evaluation_mode["uniform_without_rebalance"] == 1):
    print("--------------------Uniform Weighting without Rebalancing--------------------")
    observation = env.restart()
    done = 0
    total_return = 0
    action = [1/(len(assets)+1)] * (len(assets)+1)
    print("action:", action)
    new_state, reward, done = env.step(action)
    print("reward:", reward)
    total_return += reward
    while not done:
        action = []
        print("action:", action)
        new_state, reward, done = env.step(action)
        print("reward:", reward)
        total_return += reward
    sharpe_ratio = env.sharpe_ratio()

    return_history["uniform_without_rebalance"]= [total_return] * num_epoch
    sharpe_ratio_history["uniform_without_rebalance"] = [sharpe_ratio] * num_epoch

    print(f"------Summary: Total Return {total_return:.2f}; Sharpe Ratio {sharpe_ratio:.5f};------")

# Plotting the evaluation graphs
if not os.path.isdir("evaluation"): 
    os.makedirs("evaluation")
        
xAxis = range(1, num_epoch+1) 

plt.title("Total return over epoch")
plt.xlabel('Epoch') 
plt.ylabel('Total return')

for mode in evaluation_mode:
    if (evaluation_mode[mode] == 1):
        plt.plot(xAxis, return_history[mode], label=mode)

plt.legend()
plt.savefig("evaluation/total_return.png", dpi=300, bbox_inches="tight")
plt.clf()

plt.title("Sharpe Ratio over epoch")
plt.xlabel('Epoch') 
plt.ylabel('Sharpe Ratio')                      

for mode in evaluation_mode:
    if (evaluation_mode[mode] == 1):
        plt.plot(xAxis, sharpe_ratio_history[mode], label=mode)

plt.legend()
plt.savefig("evaluation/sharpe_ratio.png", dpi=300, bbox_inches="tight")