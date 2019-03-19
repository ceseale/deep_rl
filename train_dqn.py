#!/usr/bin/env python3
import gym
import numpy as np
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from lib import wrappers
import collections

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, n):
        indices = np.random.choice(len(self.buffer), n, replace=False)
        s, a, r, d, ns = zip(*[self.buffer[idx] for idx in indices])
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(d, dtype=np.uint8), np.array(ns)

class DQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNModel, self).__init__();

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # flatten
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.3, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.tensor(np.array([self.state])).to(device)
            q_values_v = net(state_v)
            _, action_v = torch.max(q_values_v, dim=1)
            action = int(action_v.item())

        next_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, next_state)

        self.exp_buffer.append(exp)
        self.state = next_state
        if is_done:
            self.state = self.env.reset()
            done_reward = self.total_reward
            self.total_reward = 0.0

        return done_reward

def calc_loss(batch, net, target_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    dones_v = torch.ByteTensor(dones).to(device)

    q_values_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_q_values_v = target_net(next_states_v).max(1)[0]
    next_q_values_v[dones_v] = 0.0
    next_q_values_v = next_q_values_v.detach()

    expected_q_values = next_q_values_v * GAMMA + rewards_v
    return nn.MSELoss()(q_values_v, expected_q_values)


def dqn(args):
    env = wrappers.make_env(args.env)

    net = DQNModel(env.observation_space.shape, env.action_space.n).to(args.device)
    tgt_net = DQNModel(env.observation_space.shape, env.action_space.n).to(args.device)

    writer = SummaryWriter(comment=args.exp_name)

    buffer = ExperienceBuffer(args.replay_size)
    agent = Agent(env, buffer)
    epsilon = args.epsilon_start
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    total_rewards = []
    action_count = 0
    while True:
        epsilon = max(args.epsilon_end, args.epsilon_start - action_count / args.epsilon_decay)
        reward = agent.play_step(net, epsilon, device=args.device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            writer.add_scalar("epsilon", epsilon, action_count)
            writer.add_scalar("avrage_reward", mean_reward, action_count)
            writer.add_scalar("reward", reward, action_count)
            if mean_reward > args.reward_bound:
                print("Solved in %d actions!" % action_count)
                break
        action_count += 1

        if len(buffer) < args.replay_start_size:
            continue

        if action_count % args.sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss = calc_loss(batch, net, tgt_net, device=args.device)
        loss.backward()
        optimizer.step()

    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.02)
    parser.add_argument('--epsilon_decay', type=float, default=10**5)
    parser.add_argument('--replay_start_size', type=int, default=10000)
    parser.add_argument('--sync_target_frames', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--replay_size', type=float, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--reward_bound', type=int, default=19.5)

    args = parser.parse_args()

    dqn(args)
