from ddpg.agent import Agent
import numpy as np
from env.trading_simulator import TradingSimulator
import matplotlib.pyplot as plt 
import os

# Configurations
assets = [
    "FUTU",
    "NVDA",
]
rebalance_window = 10
tx_fee_per_share = 0.005
principal=1000000
num_epoch = 5

env = TradingSimulator(principal=principal, assets=assets, start_date="2024-01-01", end_date="2024-11-11", 
                       rebalance_window=rebalance_window, tx_fee_per_share=tx_fee_per_share)

# Default alpha=0.000025, beta=0.00025, tau=0.001, batch_size=64
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4, len(assets), rebalance_window, len(assets)],
              tau=0.001, batch_size=8, n_actions=len(assets)+1)

# agent.load_models()
np.random.seed(0)

score_history = []
sharpe_ratio_history = []
for i in range(num_epoch):
    print(f"-----------------Episode {i+1}-----------------")
    observation = env.restart()
    done = 0
    score = 0
    while not done:
        action = agent.choose_action(observation)
        print("action:", action)
        new_state, reward, done = env.step(action)
        print("reward:", reward)
        agent.remember(observation, action, reward, new_state, done)
        agent.learn()
        score += reward
        observation = new_state
    score_history.append(score)
    sharpe_ratio = env.sharpe_ratio()
    sharpe_ratio_history.append(sharpe_ratio)

    # if i % 25 == 0:
    #    agent.save_models()
    print(f"------Episode {i+1} Summary: Score {score:.2f}; Sharpe Ratio {sharpe_ratio:.5f}; Trailing 100 games avg {np.mean(score_history[-100:]):.3f} ------")

# Generating evaluation graphs
if not os.path.isdir("evaluation"): 
    os.makedirs("evaluation")
        
xAxis = range(1, num_epoch+1) 

plt.title("Total return over epoch")
plt.xlabel('Epoch') 
plt.ylabel('Total return')                  
plt.plot(xAxis, score_history)
plt.savefig("evaluation/total_return.png", dpi=300, bbox_inches="tight")
plt.clf()

plt.title("Sharpe Ratio over epoch")
plt.xlabel('Epoch') 
plt.ylabel('Sharpe Ratio')                      
plt.plot(xAxis, sharpe_ratio_history)
plt.savefig("evaluation/sharpe_ratio.png", dpi=300, bbox_inches="tight")

