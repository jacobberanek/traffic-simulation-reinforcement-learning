# Traffic Signal Reinforcement Learning
COMP-6600 Group Project — Auburn University

## Team
Jacob Beranek, Logan Miller, Parker Smith

## Overview
This project explores reinforcement learning for adaptive traffic signal control. A discrete-time intersection simulator serves as the environment, where a Q-learning agent learns to minimize vehicle wait times by controlling signal phases. The agent is evaluated against two baselines: fixed-timing control and random action selection.

## Files
- `Simulator.py` — core simulation logic, traffic environment, car/queue management, reward function, and entry point
- `graphics.py` — Tkinter-based visual renderer for the intersection, traffic lights, and car queues
- `QLearningAgent.py` — tabular Q-learning agent with epsilon-greedy exploration, Bellman updates, and Q-table persistence
- `train.py` — training loop, baseline evaluation, results export, and optional visual demo
- `qtable.pkl` — saved Q-table from the most recent training run (reusable across sessions)
- `results.csv` — per-episode evaluation stats for all three modes from the most recent run

## Running

### Train and evaluate
```
python train.py
```
Trains the Q-learning agent for 500 episodes, evaluates all three modes over 20 episodes each, prints a comparison table, and saves results to `results.csv`.

### Skip training and reuse a saved Q-table
```
python train.py --skip-train
```
Loads `qtable.pkl` and runs evaluation only. Useful if you have already trained and just want to re-evaluate.

### Watch the trained agent live
```
python train.py --visual
```
Launches a Tkinter window showing the trained agent controlling the intersection after evaluation completes.

### Run the simulator directly
```
python Simulator.py
```
Runs a single episode with the visual renderer and no agent (fixed-timing baseline).

## How It Works

### Simulator
The simulator models a four-way intersection with discrete time ticks. Cars arrive stochastically every 3 ticks (1–3 cars per batch) from a randomly chosen direction. The light cycles through NS_GREEN → NS_YELLOW → EW_GREEN → EW_YELLOW, with a configurable yellow phase delay (default: 3 ticks). A `max_green_ticks` parameter (default: 30) forces a phase switch if the agent holds a green phase too long, preventing any direction from being starved indefinitely.

### State Space
The agent observes a tuple of five values each tick:
- Bucketed queue lengths for North, South, East, and West (0=empty, 1=light, 2=moderate, 3=heavy)
- Current light state (NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW)

This gives a state space of 4^4 × 4 = 1,024 possible states, kept small intentionally for tabular Q-learning.

### Actions
- `KEEP` — hold the current light phase for another tick
- `SWITCH` — request a phase change via `trigger_light_switch()`

### Reward Function
The reward signal combines three components:
- **Queue penalty** — negative total cars waiting across all directions
- **Switch bonus** — positive reward proportional to the waiting direction's queue length when switching, encouraging the agent to switch only when it actually helps
- **Imbalance penalty** — penalises holding a green phase while the opposing direction has a significantly longer queue

### Baselines
- **Fixed timing** — switches every 10 ticks regardless of queue state
- **Random** — picks KEEP or SWITCH randomly each tick

Q-learning matches fixed timing closely on throughput and substantially outperforms random. Under uniform random traffic, fixed timing is a strong baseline; Q-learning's advantage is expected to be more pronounced under asymmetric traffic conditions where one direction receives significantly more cars.

## Dependencies
Python 3.10+, tkinter (standard library). No additional installs required.

## Future Work
- **Asymmetric traffic** — the current simulation uses uniform random arrival rates across all directions. A weighted spawn configuration (e.g. N/S receiving 4x more cars than E/W) would better reflect real-world conditions and is expected to show a larger advantage for the Q-learning agent over fixed timing, since fixed timing cannot adapt its green phase duration to uneven demand.