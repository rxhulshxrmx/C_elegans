# C. elegans
Studying neural network learning in C. elegans sensorimotor control.

## Overview

This project combines MuJoCo physics simulation with RNN training to study how neural networks learn to control worm behavior. We train RNNs for 1D and 3D navigation tasks and analyze how learning changes network weights and neuron correlations.

## Quick Start

```bash
# 1. Train RNNs and analyze neural changes
python3 simple_rnn.py

# 2. View MuJoCo worm simulation
mjpython mujoco_viewer.py

# 3. Check generated analysis plots
ls results/
```

## Requirements

- Python 3.9+
- MuJoCo physics engine
- Required packages: `pip install "gym[mujoco]" mujoco matplotlib`

## Running the Simulation

```bash
mjpython mujoco_viewer.py
```

Choose option 1 for the full 25-segment C. elegans with chemotaxis behavior, or option 2 for a simple 3-segment swimmer.

The worm uses chemotaxis to navigate toward the reward, demonstrating realistic C. elegans behavior. Console shows distance to reward and concentration levels.


## Technical Details

The worm model includes:
- 26 rigid bodies (head + 25 segments)
- 27 joints for articulation
- 24 actuators for muscle control
- Sinusoidal swimming pattern with chemotaxis modulation

## RNN Training and Analysis

**Goal**: Train RNN networks to control C. elegans agents and analyze weight/correlation changes.

### Scripts Available

- `simple_rnn.py` - **Main research script** demonstrating RNN training and analysis
- `train_rnn_elegans.py` - Full-featured training pipeline with behavioral cloning

### Research Pipeline

```bash
python3 simple_rnn.py
```

This script:
1. Trains RNN networks for 1D and 3D navigation tasks
2. Extracts network weights before and after training
3. Analyzes weight changes and neuron correlations
4. Generates comprehensive analysis visualizations
5. Compares results between 1D and 3D scenarios

### Results Generated

- Weight change analysis (magnitude and relative changes)
- Neuron correlation matrices (initial vs final)
- Training loss curves
- Comparative analysis plots saved as PNG files


## Research Results

When you run the research scripts, analysis plots are automatically saved to `results/`:
- `simple_rnn_1d.png` - Complete 1D navigation analysis
- `simple_rnn_3d.png` - Complete 3D navigation analysis

Each plot shows:
- Training loss curves (learning progress)
- Weight change magnitudes (neural reorganization)
- Correlation matrices (before/after learning)
- Difference maps (learning-induced changes)
