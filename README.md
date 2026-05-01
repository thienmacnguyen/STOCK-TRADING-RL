# STOCK-TRADING-RL
# Stock Trading Simulation using Reinforcement Learning

## Overview
This project simulates stock trading using Reinforcement Learning to optimize investment strategies based on historical stock data from QCOM (NASDAQ).
The system uses a Deep Q-Network (DQN) to learn buy, sell, and hold decisions through interaction with a custom trading environment.
---
## Features
- Custom trading environment
- Deep Q-Network (DQN) implementation
- Time-series feature engineering
- Data preprocessing and normalization
- Performance comparison with buy-and-hold strategy

## Dataset
The dataset is split into:
- Train set (train.xlsx)
- Validation set (val.xlsx)
- Test set (test.xlsx)

## How to run
pip install -r requirements.txt
python train.py
python test.py

## Results
- Achieved better cumulative return than the buy-and-hold baseline during simulation
- Demonstrated consistent learning behavior across multiple training episodes
- Processed 3,000+ trading sessions (QCOM dataset from NASDAQ)

## Tech Stack
- Python
- PyTorch
- Pandas, NumPy
