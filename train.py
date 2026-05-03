from agent import Agent
from utils import getState, getStockData, formatPrice

window_size = 10
episode_count = 10
batch_size = 32

data = getStockData("data/train.xlsx")
l = len(data) - 1

agent = Agent(window_size)

for e in range(episode_count):
    print(f"\nEpisode {e+1}/{episode_count}")
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(data[t])

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            profit = data[t] - bought_price
            reward = max(profit, 0)
            total_profit += profit

        done = (t == l - 1)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

        if done:
            print(f"Total Profit: {formatPrice(total_profit)}")

agent.model.save("model.h5")
print("\nModel saved!")