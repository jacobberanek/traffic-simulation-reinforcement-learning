# Traffic Signal Reinforcement Learning
COMP-6600 Group Project — Auburn University

## Team
Jacob Beranek, Logan Miller, Parker Smith

## Overview
This project explores reinforcement learning for adaptive traffic signal control. A discrete-time intersection simulator serves as the environment, where an RL agent will learn to minimize vehicle wait times by controlling signal phases.


## Files
- `simulator.py` — core simulation logic, traffic environment, and entry point
- `graphics.py` — Tkinter-based visual renderer

## Running
```
python simulator.py
```
Toggle `visual=True/False` on the last line of `simulator.py` to run with or without the graphics window.

## Dependencies
Python 3.x, tkinter (standard library), no additional installs required.
