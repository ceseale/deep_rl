#!/usr/bin/env python3
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

Exp = namedtuple('Experience', ['reward', 'action', 'state', 'done'])

class Buffer:
    def __init__(self):
        self.trajectories = []
        self.states = []
        self.seen = 0

    def store(self, reward, action, state, done):
        self.states.append(state)
        self.trajectories.append(Exp(reward = reward, action=action, state=state, done=done))
        self.seen += 1

    def __len__(self):
        return len(self.trajectories)

    def clear(self):
        self.states.clear()
        self.trajectories.clear()

class ReinforceNet(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ReinforceNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class Agent():
    def __init__(self, net):
        self.net = net
        self.sm = F.softmax

    def act(self, obs_v):
        action_logits_v = net(obs_v)
        probs = self.sm(action_logits_v, dim=1).squeeze().detach().numpy()
        action = np.random.choice(len(probs), p=probs)
        return action

def cacl_qvals(rewards, gamma):
    rewards_sum = 0
    vals = []
    for reward in reversed(rewards):
        rewards_sum = reward + gamma * rewards_sum
        vals.append(rewards_sum)
    vals = list(reversed(vals))
    # using mean baseline
    mean = np.mean(vals)
    return [q - mean for q in vals]

def reinforce(args):
    env = gym.make(args.env)
    writer = SummaryWriter(comment=args.exp_name)
    net = ReinforceNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    agent = Agent(net)
    buffer = Buffer()
    obs = env.reset()
    episode_count = 0
    action_batch, state_batch, qvals_batch, total_rewards  = [], [], [], []

    while True:
        obs_v = torch.FloatTensor([obs])
        action = agent.act(obs_v)
        next_obs, reward, is_done, _ = env.step(action)
        buffer.store(reward, action, obs, is_done)
        total_rewards.append(reward)
        obs = next_obs
        if is_done:
            qvals_batch.extend(cacl_qvals([tr.reward for tr in buffer.trajectories], args.gamma))
            action_batch.extend([tr.action for tr in buffer.trajectories])
            state_batch.extend([tr.state for tr in buffer.trajectories])
            buffer.clear()
            obs = env.reset()
            writer.add_scalar("avg_reward", np.mean(total_rewards[:100]), episode_count)
            episode_count += 1

        if episode_count < args.episodes_wait:
            continue

        states_v = torch.FloatTensor(state_batch)
        batch_actions_v = torch.LongTensor(action_batch)
        batch_qvals_v = torch.FloatTensor(qvals_batch)

        optimizer.zero_grad()
        action_logprobs_v = F.log_softmax(net(states_v), dim=1)
        pg_v = batch_qvals_v * action_logprobs_v[range(len(action_logprobs_v)), batch_actions_v]
        loss_v = -pg_v.mean()
        loss_v.backward()
        optimizer.step()
        batch_episodes = 0
        buffer.clear()

    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='reinforce')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--episodes_wait', type=int, default=4)

    args = parser.parse_args()

    ppo(args)

