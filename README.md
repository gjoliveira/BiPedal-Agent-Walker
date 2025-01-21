# BiPedal-Agent-Walker
Developed a reinforcement learning agent using an existing OpenAI Gym / Gymnasium environment as a base.


_______________________________________________________________________________________________________

This project focuses on customizing the OpenAI Gym's Bipedal Walker environment and implementing reinforcement learning (RL) agents using the Stable-Baselines3 library. The primary goal is to explore the impact of various environment customizations, reward structures, and hyperparameter tuning on agent performance.

# Project Overview

### Environment Customization:

Observations: The agent perceives a fully observable state space with 24 continuous variables, including hull dynamics, joint angles, ground contact sensors, and LIDAR-based terrain data.

Actions: Continuous control of motor speeds for four joints (hips and knees) within the range [-1, 1].

Rewards: Incentivizing forward movement, overcoming obstacles, and lifting legs, while penalizing instability, vertical falls, and inactivity.

### Key Experiments:

##### Reward Design:

Various reward strategies were tested, such as alternating leg use, overcoming steep terrain, and minimizing instability.

Penalty schemes for falling, torso inclination, and prolonged inactivity were implemented.

##### Agent Customization:

Added ankles to the agent (creating "feet") to assess their impact on performance.

##### Hyperparameter Tuning:

Explored the effect of learning rate, batch size, and discount factor on TRPO, showing significant performance improvements.

##### Comparison Across RL Algorithms:

Evaluated PPO, SAC, and TRPO, identifying algorithm-specific behaviors and performance.

# Key Findings:

The inclusion of feet did not yield significant performance benefits and, in some cases, hindered agent performance.

Hyperparameter tuning significantly improved performance, even with fewer training timesteps.

Reward structures emphasizing forward movement were critical for overcoming terrain challenges.

# Results and Insights:

The project demonstrated the importance of tailored reward functions and hyperparameter optimization in enhancing the capabilities of RL agents. Comparative analysis revealed that while certain environment modifications (like feet) were less effective, properly tuned agents performed significantly better in challenging scenarios.





https://github.com/user-attachments/assets/0e20d54a-4e7c-48a5-82bf-54b21b92e646 [ Control- SAC with no rewards]





