from agent import Agent
from utils import getState, getStockData, formatPrice
import numpy as np

window_size = 10
data = getStockData("data/test.csv")
l = len(data) - 1

agent = Agent(window_size)
agent.model.load_weights("model.h5")
agent.epsilon = 0

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

profits = []
trades = 0
wins = 0

print("----- TESTING -----")

for t in range(l):
    action = agent.act(state)
    next_state = getState(data, t + 1, window_size + 1)

    if action == 1:
        agent.inventory.append(data[t])

    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        profit = data[t] - bought_price
        total_profit += profit
        profits.append(profit)

        trades += 1
        if profit > 0:
            wins += 1

        print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(profit)}")

    state = next_state

print("\n----- RESULT -----")
print(f"Total Profit: {formatPrice(total_profit)}")

if trades > 0:
    print(f"Win Rate: {wins/trades:.2f}")

if len(profits) > 1:
    sharpe = np.mean(profits) / (np.std(profits) + 1e-6)
    print(f"Sharpe Ratio: {sharpe:.4f}")
