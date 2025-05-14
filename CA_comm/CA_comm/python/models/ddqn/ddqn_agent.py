import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----- Q-Network Definition -----
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- Replay Buffer -----
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ----- DDQN Agent Training -----
def train_ddqn(args):
    # Determine directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.abspath(os.path.join(script_dir, os.pardir, 'data'))

    # Placeholder for environment: should implement reset() and step(action)
    from simulator_env import SimulatorEnv  # custom env wrapping MATLAB
    env = SimulatorEnv()

    # Dimensions
    state_dim  = env.state_size
    action_dim = env.action_size

    # Networks
    q_net      = QNetwork(state_dim, action_dim, args.hidden_dim).to(args.device)
    target_net = QNetwork(state_dim, action_dim, args.hidden_dim).to(args.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer  = optim.Adam(q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer     = ReplayBuffer(args.buffer_size)

    # Visualization lists
    rewards_list = []
    epsilons_list = []

    epsilon = args.epsilon_start
    for episode in range(1, args.episodes+1):
        state = env.reset()
        total_reward = 0
        for t in range(args.max_steps):
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(args.device)
                    q_values = q_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Update
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                update_q_network(q_net, target_net, optimizer, batch, args)

            if done:
                break

        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # Update target network
        if episode % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Log and collect metrics
        print(f"Episode {episode}/{args.episodes} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")
        rewards_list.append(total_reward)
        epsilons_list.append(epsilon)
    #colab 추가
    import pandas as pd
    df = pd.DataFrame({
        'reward':  rewards_list,
        'epsilon': epsilons_list
    })
    df.to_csv('train_history.csv', index=False)
    print("▶ train_history.csv 저장 완료")

    # Save trained model
    model_path = os.path.join(script_dir, args.out_model)
    torch.save(q_net.state_dict(), model_path)
    print(f"DDQN model saved to {model_path}")

    # Plot reward curve
    plt.figure()
    plt.plot(range(1, args.episodes+1), rewards_list, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DDQN Reward Curve')
    plt.legend()
    plt.savefig(os.path.join(script_dir, 'ddqn_reward_curve.png'))
    print("Reward curve saved to ddqn_reward_curve.png")

    # Plot epsilon decay curve
    plt.figure()
    plt.plot(range(1, args.episodes+1), epsilons_list, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Curve')
    plt.legend()
    plt.savefig(os.path.join(script_dir, 'ddqn_epsilon_curve.png'))
    print("Epsilon decay curve saved to ddqn_epsilon_curve.png")

# ----- Training Step -----
def update_q_network(q_net, target_net, optimizer, batch, args):
    states, actions, rewards, next_states, dones = batch
    states      = torch.FloatTensor(states).to(args.device)
    actions     = torch.LongTensor(actions).unsqueeze(1).to(args.device)
    rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(args.device)
    next_states = torch.FloatTensor(next_states).to(args.device)
    dones       = torch.FloatTensor(dones).unsqueeze(1).to(args.device)

    # Current Q
    q_values = q_net(states).gather(1, actions)
    # Double Q-learning target
    with torch.no_grad():
        next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        next_q       = target_net(next_states).gather(1, next_actions)
        target_q     = rewards + (1 - dones) * args.gamma * next_q

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----- CLI Entry Point -----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDQN agent for CC scheduling')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--out_model', type=str, default='ddqn_ccmask.pth')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    train_ddqn(args)
