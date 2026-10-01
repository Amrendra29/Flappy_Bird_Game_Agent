# 🐦 Flappy Bird AI Agent — Deep Q-Learning

An AI agent that learns to play **Flappy Bird autonomously using Deep Reinforcement Learning (DQN)**.

Instead of manually defining when the bird should flap, the agent learns an effective strategy through **trial and error** by interacting with the game environment, receiving rewards, storing experiences, and updating a neural network.

## 🚀 Project Overview

This project implements a **Deep Q-Network (DQN)** agent for the Flappy Bird game.

The agent observes the current game state and uses a neural network to estimate the **Q-value of each possible action**. It then selects an action based on the learned Q-values and improves its behavior through repeated interaction with the environment.

### Agent Workflow

```text
Game State
    ↓
DQN Neural Network
    ↓
Q-Values for Actions
    ↓
Action Selection
    ↓
Flappy Bird Environment
    ↓
Reward + Next State
    ↓
Experience Replay
    ↓
DQN Training
```

## 🧠 Technologies Used

- **Python**
- **PyTorch**
- **Gymnasium**
- **NumPy**
- **Deep Q-Network (DQN)**
- **Experience Replay**
- **Target Network**
- **YAML** for hyperparameter configuration
- **Git & GitHub**

## 🏗️ Project Architecture

The project is divided into multiple components:

```text
Flappy_Bird/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── game_flappy_bird.py
├── parameters.yaml
│
├── runs/
│   ├── flappybirdv0.log
│   └── flappybirdv0.pt
│
└── README.md
```

### 📄 File Description

| File | Description |
|------|-------------|
| `agent.py` | Main RL agent, training loop and optimization |
| `dqn.py` | Deep Q-Network architecture |
| `experience_replay.py` | Stores and samples past experiences |
| `game_flappy_bird.py` | Flappy Bird environment |
| `parameters.yaml` | Hyperparameters and configuration |
| `runs/` | Training logs and saved model checkpoints |

## 🎮 Actions

The agent has two possible actions:

| Action | Description |
|--------|-------------|
| `0` | Do Nothing |
| `1` | Flap |

The DQN learns which action is better based on the current state of the game.

## 📊 Deep Q-Learning

The agent uses a **Deep Q-Network** to approximate the Q-function:

```text
Q(state, action)
```

For a given game state, the neural network outputs Q-values for the available actions.

```text
              Game State
                   │
                   ▼
          ┌─────────────────┐
          │   DQN Network   │
          └────────┬────────┘
                   │
             Q-Values
              /       \
             /         \
            ▼           ▼
       Do Nothing      Flap
```

The action with the highest estimated Q-value can be selected by the agent.

## 🔄 Experience Replay

The project uses an **Experience Replay** mechanism.

During gameplay, experiences are collected in the form:

```text
(state, action, reward, next_state, termination)
```

These experiences are stored in replay memory.

During training, random batches are sampled from the replay buffer rather than learning only from the most recent experience.

This helps reduce the correlation between consecutive experiences and improves training stability.

## 🎯 Target Network

The implementation also uses a separate **target DQN**.

The target Q-value is calculated using the target network:

```python
target_q = rewards + \
    (1 - terminations) * gamma * \
    target_dqn(next_states).max(dim=1)[0]
```

The current Q-value is obtained from the policy network:

```python
current_q = policy_dqn(states).gather(
    dim=1,
    index=actions.unsqueeze(dim=1)
).squeeze()
```

The loss is then calculated between the predicted Q-value and target Q-value.

```text
Target Q-Value
      │
      ▼
   Loss Function
      ▲
      │
Current Q-Value
```

The policy network is updated using backpropagation and the optimizer.

## 🧮 Q-Learning Update

The target value follows the standard DQN formulation:

```text
Target Q =
Reward + γ × max Q(next_state, next_action)
```

For terminal states:

```text
Target Q = Reward
```

where:

- `Reward` = reward received from the environment
- `γ` = discount factor
- `Q` = estimated action value

## 🔥 Training Process

The agent follows this training loop:

```text
1. Initialize environment
        ↓
2. Initialize policy DQN
        ↓
3. Initialize target DQN
        ↓
4. Observe game state
        ↓
5. Select action
        ↓
6. Execute action
        ↓
7. Receive reward & next state
        ↓
8. Store experience
        ↓
9. Sample mini-batch
        ↓
10. Calculate target Q-values
        ↓
11. Calculate current Q-values
        ↓
12. Calculate loss
        ↓
13. Backpropagation
        ↓
14. Update neural network
        ↓
15. Repeat
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Amrendra29/Flappy_Bird_Game_Agent.git
cd Flappy_Bird_Game_Agent
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Train the Agent

Run:

```bash
python agent.py flappybirdv0 --train
```

During training, the agent interacts with the environment and continuously updates its DQN using experiences collected from gameplay.

Training information is stored in the `runs/` directory.

## 🎯 Run the Trained Agent

After training, run:

```bash
python agent.py flappybirdv0
```

The saved model is loaded and the trained agent plays Flappy Bird automatically.

## 📈 Learning Process

At the beginning of training, the agent has little knowledge about the environment.

```text
Random / Poor Decisions
          ↓
More Game Experiences
          ↓
Experience Replay
          ↓
DQN Updates
          ↓
Better Q-Value Estimates
          ↓
Improved Action Selection
          ↓
Longer Survival
          ↓
Better Performance
```

The agent gradually learns when it should flap to avoid obstacles and continue surviving.

## 🧪 Model Training

The project stores the trained model/checkpoint in the `runs` directory.

Example:

```text
runs/
├── flappybirdv0.log
└── flappybirdv0.pt
```

The `.pt` file contains the trained PyTorch model/checkpoint.

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of:

- Reinforcement Learning
- Deep Q-Learning
- Deep Neural Networks
- Q-Learning
- Experience Replay
- Target Networks
- Bellman Equation
- Reward-based learning
- Exploration vs. exploitation
- State and action representation
- PyTorch model optimization
- Backpropagation
- Gymnasium environments
- Model checkpointing
- Hyperparameter configuration

## 📌 Challenges

Some of the challenges involved in developing this project include:

- Designing an appropriate state representation
- Designing meaningful rewards
- Stabilizing DQN training
- Managing exploration and exploitation
- Implementing experience replay
- Maintaining a separate target network
- Handling environment observation issues
- Debugging training and model checkpoints
- Improving the agent's gameplay performance

## 🔮 Future Improvements

Possible improvements include:

- Implement **Double DQN**
- Implement **Dueling DQN**
- Improve reward shaping
- Tune hyperparameters
- Add training-performance graphs
- Track episode rewards and scores
- Compare different DQN architectures
- Improve exploration strategy
- Add automated evaluation of trained models

## 🏆 Project Outcome

This project demonstrates how **Deep Reinforcement Learning can be used to train an autonomous game-playing agent**.

Rather than hard-coding the game strategy, the agent learns through interaction with the environment and improves its decisions using **Deep Q-Learning, Experience Replay, and a Target Network**.

The project provided practical experience with the complete RL pipeline:

```text
Environment
     ↓
State
     ↓
Action
     ↓
Reward
     ↓
Experience Replay
     ↓
DQN
     ↓
Optimization
     ↓
Improved Agent
```

## 👨‍💻 Author

**Amrendra Singh**

B.Tech — Computer Science & Engineering (Artificial Intelligence)


---

⭐ If you found this project interesting, consider giving the repository a star!
