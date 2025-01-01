from ddpg.agent import Agent
import numpy as np
from env.trading_simulator import TradingSimulator

# from utils import plotLearning

# config
assets = [
    "FUTU",
    "NVDA",
]
rebalance_window = 10


env = TradingSimulator(
    1000000,
    assets=assets,
    start_date="2024-01-01",
    end_date="2024-11-11",
    rebalance_window=rebalance_window,
)
agent = Agent(
    alpha=0.000025,
    beta=0.00025,
    input_dims=[4, len(assets), rebalance_window, len(assets)],
    tau=0.001,
    batch_size=16,
    layer1_size=400,
    layer2_size=300,
    n_actions=len(assets),
)
# agent.load_models()
np.random.seed(0)

score_history = []
for i in range(5):
    print(f"-----------------Episode {i}-----------------")
    init_holdings, input_tensor = env.restart()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(input_tensor)
        print("action: ", action)
        (
            old_holdings,
            old_input_tensor,
            action,
            reward,
            new_holdings,
            new_input_tensor,
            done,
        ) = env.step(action)
        agent.remember(
            # old_holdings,
            old_input_tensor,
            action,
            reward,
            # new_holdings,
            new_input_tensor,
            done,
        )
        agent.learn()
        score += reward
        observation = new_input_tensor
    score_history.append(score)

    # if i % 25 == 0:
    #    agent.save_models()
    print(
        f"------Episode {i}; Score {score:.2f}; Trailing 100 games avg {np.mean(score_history[-100:]):.3f} ------"
    )


filename = "LunarLander-alpha000025-beta00025-400-300.png"
# plotLearning(score_history, filename, window=100)
