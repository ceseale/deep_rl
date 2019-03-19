#!/usr/bin/env python3
import torch
from torch import nn
from collections import namedtuple, deque
import numpy as np
import gym
import roboschool
import torch.optim as optim
import math
from tensorboardX import SummaryWriter
import sys
import time
import operator
from datetime import timedelta

"""
Buffer to hold trajectories
"""

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

    def clear(self):
        self.states = []
        self.trajectories = []

class CriticModel(nn.Module):
    def __init__(self, observation_size, hidden_size = 64):
        super(CriticModel, self).__init__()
        self.val = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.val(x)

class ActorModel(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size = 64):
        super(ActorModel, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )

        self.logstd = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        return self.mu(x)

class Agent:
    def __init__(self, net, device='cpu'):
        self.device = device
        self.net = net

    def __call__(self, states):
        states_v = torch.tensor(states).to(self.device)
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.cpu().detach().numpy()
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        actions = np.clip(actions, -1, 1)
        return actions

def calc_adv(trajectories, states_v, net, gamma, gae_lambda, device='cpu'):
    values_v = net(states_v)
    values = values_v.data.cpu().numpy()
    last_gae = 0
    advs = []
    refs = []
    for exp, val, next_val in zip(reversed(trajectories[:-1]), reversed(values[:-1]), reversed(values[1:])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        advs.append(last_gae)
        refs.append(last_gae + val)

    advs_v = torch.FloatTensor(list(reversed(advs))).to(device)
    refs_v = torch.FloatTensor(list(reversed(refs))).to(device)

    return advs_v, refs_v

def calc_logprobs(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))

    return p1 + p2

def test_net(net, env, count = 10, device = "cpu"):
    reward = 0.0
    steps = 0
    for i in range(count):
        obs = env.reset()
        while True:
            obs_v = torch.FloatTensor(obs).to(device)
            mu = net(obs_v).squeeze(-1).data.cpu().numpy()
            action = np.clip(mu, -1, 1)
            obs_v, r, d, _ = env.step(action)
            reward += r
            steps += 1
            if d:
                obs = env.reset()
                break

    return reward / count, steps / count

def ppo(args):
    env = gym.make(args.env)
    observation_size = env.observation_space.shape[0]
    actions_size = env.action_space.shape[0]
    critic_model = CriticModel(observation_size).to(args.device)
    actor_model = ActorModel(observation_size, actions_size).to(args.device)
    print(actor_model)
    print(critic_model)
    agnet = Agent(actor_model, args.device)
    exp_buffer = Buffer()
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=args.critic_learning_rate)
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=args.actor_learning_rate)
    count_steps = 0

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    writer = SummaryWriter(comment=args.exp_name)
    reward_100_data = []

    # runnig for up to 1000 perhaps stoping before
    for _ in range(1000):
        for trajectory_index in range(args.trajectory_size):
            action = agnet(o)
            o, r, d, _ = env.step(action)
            exp_buffer.store(r, action, o, d)
            reward_100_data.append(r)
            print("mean_reward_100 {}".format(np.mean(reward_100_data)))

            if d:
                o = env.reset()

        # GAE
        states_v = torch.FloatTensor(exp_buffer.states)
        advs_v, refs_v = calc_adv(exp_buffer.trajectories, states_v, critic_model, args.gamma, args.gae_lambda, args.device)
        advs_v = (advs_v - torch.mean(advs_v)) / torch.std(advs_v)
        # get logprobs for actions
        actions = np.array([t.action for t in exp_buffer.trajectories])
        actions_v = torch.FloatTensor(actions).to(args.device)
        mu_v = actor_model(states_v)
        old_logprobs = calc_logprobs(mu_v, actor_model.logstd, actions_v).detach()

        trajectory = exp_buffer.trajectories[:-1]
        states_v = states_v[:-1]

        mean_reward, mean_steps = test_net(actor_model, env, 10, args.device)

        print("mean_reward {}".format(mean_reward))
        print("mean_steps {}".format(mean_steps))
        print("seen {}".format(exp_buffer.seen))

        writer.add_scalar("mean_reward", mean_reward, exp_buffer.seen)
        writer.add_scalar("mean_steps", mean_steps, exp_buffer.seen)

        sum_loss_value = 0.0
        sum_loss_policy = 0.0

        for epoch in range(args.ppo_epoches):
            for batch_index in range(0, len(trajectory), args.ppo_batch_size):
                states_batch_v = states_v[batch_index:batch_index + args.ppo_batch_size]
                refs_batch_v = refs_v[batch_index:batch_index + args.ppo_batch_size]
                actions_batch_v = actions_v[batch_index:batch_index + args.ppo_batch_size]
                old_logprobs_batch = old_logprobs[batch_index:batch_index + args.ppo_batch_size]
                advs_batch_v = advs_v[batch_index:batch_index + args.ppo_batch_size]

                critic_optimizer.zero_grad()
                values_v = critic_model(states_batch_v)
                loss_ob = torch.nn.functional.mse_loss(values_v.squeeze(-1), refs_batch_v.squeeze(-1))
                loss_ob.backward()
                critic_optimizer.step()

                actor_optimizer.zero_grad()
                mu_v = actor_model(states_batch_v)
                logprobs_batch = calc_logprobs(mu_v, actor_model.logstd, actions_batch_v)
                ratio = torch.exp(logprobs_batch - old_logprobs_batch)
                kl = torch.mean(old_logprobs_batch - logprobs_batch)
                entropy = torch.mean(-logprobs_batch)

                surr = advs_batch_v * ratio
                clipped = advs_batch_v * torch.clamp(ratio, 1.0 - args.ppo_eps, 1.0 + args.ppo_eps)
                loss = -torch.min(surr, clipped).mean()
                loss.backward()
                actor_optimizer.step()

                count_steps += 1
                writer.add_scalar("kl", kl, count_steps)
                writer.add_scalar("entropy", entropy, count_steps)

        print("count_steps {}".format(count_steps))
        writer.add_scalar("values", refs_v.mean().item(), exp_buffer.seen)

        exp_buffer.clear()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--env', type=str, default='RoboschoolHalfCheetah-v1')
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ppo_epoches', type=int, default=10)
    parser.add_argument('--actor_learning_rate', type=float, default=1e-4)
    parser.add_argument('--critic_learning_rate', type=float, default=1e-3)
    parser.add_argument('--ppo_batch_size', type=int, default=64)
    parser.add_argument('--trajectory_size', type=int, default=2113)
    parser.add_argument('--ppo_eps', type=float, default=0.2)

    args = parser.parse_args()

    ppo(args)


