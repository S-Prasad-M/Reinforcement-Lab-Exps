# Reinforcement Learning Lab Experiments

This repository contains a series of reinforcement learning experiments implemented in Jupyter Notebooks. Each lab explores different RL algorithms and concepts to solve specific problems.

---

## Lab 2: Multi-Armed Bandits for Ad Optimization

Focuses on solving the multi-armed bandit problem using various strategies to maximize click-through rates (CTR) in an ad-serving simulation.

- **Notebook:** [`RL_Lab2_Work.ipynb`](RL_Lab2_Work.ipynb)
- **Core Algorithm:** Epsilon-Greedy, UCB, Softmax
- **Tasks:**
    1.  **Contextual Bandits:** Implemented a contextual ε-Greedy agent that adapts its ad selection based on user segments (e.g., teenagers, adults) with varying preferences.
    2.  **Budget Constraints:** Added per-ad budget limitations, forcing the agent to learn within spending constraints.
    3.  **CTR Drift Comparison:** Compared the performance of ε-Greedy, UCB, and Softmax agents in a non-stationary environment where ad CTRs change over time.

---

## Lab 3: Warehouse Robot Navigation with MDPs

Models a warehouse environment as a Markov Decision Process (MDP) and uses Value Iteration to find the optimal path for a delivery robot.

- **Notebook:** [`RL_Lab3_Work.ipynb`](RL_Lab3_Work.ipynb)
- **Core Algorithm:** Value Iteration
- **Tasks:**
    1.  **Directional Constraints:** Introduced one-way paths, requiring the robot to learn policies that respect movement restrictions.
    2.  **Dynamic Goals:** The delivery target changes periodically, testing the agent's ability to adapt its policy to new objectives.
    3.  **Time-Dependent Penalties:** Simulated human worker presence by adding time-varying penalties to certain zones, forcing the agent to learn a time-aware policy.

---

## Lab 4: Smart Ambulance Dispatch with Monte Carlo

Uses Monte Carlo methods to train an agent for smart ambulance dispatch in a dynamic and probabilistic urban grid environment.

- **Notebook:** [`RL_Lab4_Work.ipynb`](RL_Lab4_Work.ipynb)
- **Core Algorithm:** Monte Carlo Control
- **Tasks:**
    1.  **Dynamic Hospital Queues:** The agent learns to select the optimal hospital based on both travel time and real-time bed availability, with penalties for delays.
    2.  **Mid-Episode Obstacles:** The environment introduces sudden roadblocks, requiring the agent to find alternative routes during an episode.
    3.  **Time-Varying Occupancy:** Hospital availability fluctuates, and the agent learns a policy to choose the least crowded destination.

---

## Lab 5: Elevator Control with Temporal Difference Learning

Applies Temporal Difference (TD) learning methods (Q-Learning and SARSA) to develop an optimal control policy for an elevator in a smart building.

- **Notebook:** [`RL_Lab5_Work.ipynb`](RL_Lab5_Work.ipynb)
- **Core Algorithms:** Q-Learning, SARSA
- **Tasks:**
    1.  **Hyperparameter Tuning:** Analyzed the impact of varying exploration rates (ε) and discount factors (γ) on the agent's learning stability and performance.
    2.  **Energy Consumption Simulation:** Penalized unnecessary elevator movements and direction changes to encourage an energy-efficient policy.
    3.  **Time-Based Demand:** Modeled peak hours by introducing time-dependent request patterns, training the agent to adapt its strategy to fluctuating demand.

---

## Lab 6: Rescue Robot with N-Step TD and Advanced Techniques

Implements n-step TD learning for a rescue robot and explores more advanced concepts like multi-agent coordination and dynamic hazard avoidance.

- **Notebook:** [`RL_Lab6_Work.ipynb`](RL_Lab6_Work.ipynb)
- **Core Algorithms:** N-Step TD, TD(λ), Q-Learning
- **Tasks:**
    1.  **Multi-Robot Coordination:** Deployed two robots that learn collaboratively using a shared Q-table, with mechanisms to avoid collisions and redundant rescues.
    2.  **TD(λ) with Eligibility Traces:** Implemented TD(λ) to allow rewards to propagate more efficiently back through a trajectory, improving learning speed.
    3.  **Moving Traps:** Introduced dynamic traps that change position, forcing the agent to learn in a non-stationary environment.
    4.  **Hazard Prediction:** The agent learns to predict and avoid high-risk zones by maintaining a `hazard_map` based on past encounters with traps.

---

## Lab 7: Drone Navigation with Dyna-Q

Uses the Dyna-Q algorithm, which combines model-free and model-based learning, to train a drone to navigate an urban grid with obstacles.

- **Notebook:** [`RL_Lab7_Work.ipynb`](RL_Lab7_Work.ipynb)
- **Core Algorithm:** Dyna-Q
- **Problem:** A delivery drone must find the shortest path from a warehouse to a destination, learning from both real interactions and simulated experiences from its learned world model. The agent performs planning steps to update its policy even when not moving in the real world.