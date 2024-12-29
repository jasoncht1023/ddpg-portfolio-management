from ddpg.agent import Agent
import numpy as np
from env.trading_simulator import TradingSimulator
# from utils import plotLearning

assets = [
    "FUTU",
    "NVDA",
]

env = TradingSimulator(1000000, assets=assets, start_date="2024-01-01", end_date="2024-11-11", rebalance_window=10)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[len(assets)*5], tau=0.001,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(5):
    observation = env.restart()
    print("observation:", observation)
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        print("action: ", action)
        new_state, reward, done = env.step(action)
        print("new_state, reward, done: ", new_state, reward, done)
        agent.remember(observation, action, reward, new_state, int(done))
        agent.learn()
        score += reward
        observation = new_state
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
# plotLearning(score_history, filename, window=100)
