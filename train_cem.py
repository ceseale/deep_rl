#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

class CEMNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(CEMNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def cem(args):
    env = gym.make(args.env)
    writer = SummaryWriter(comment=args.exp_name)
    Episode = namedtuple('Episode', field_names=['reward', 'steps'])
    EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
    total_reward = 0.0
    episodes = []
    action_states = []
    obs = env.reset()

    Episode = namedtuple('Episode', field_names=['reward', 'state_action_pair'])
    StateActionPair = namedtuple('StateActionPair', field_names=['observation', 'action'])

    cem_net = CEMNet(env.observation_space.shape[0], args.hidden_size, env.action_space.n)
    optimizer = optim.Adam(params=cem_net.parameters(), lr=0.01)
    ce_loss = nn.CrossEntropyLoss()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        actions_scores_v = cem_net(obs_v)

        action_probs = sm(actions_scores_v).detach().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        next_obs, reward, is_done, _ = env.step(action)
        action_states.append(StateActionPair(obs, action))
        total_reward += reward
        if is_done:
            episodes.append(Episode(total_reward, action_states))
            action_states = []
            total_reward = 0.0
            obs = env.reset()

        if len(episodes) < args.batch_size:
            continue

        rewards = np.array([ep.reward for ep in episodes])
        mean_reward = float(np.mean(rewards))
        reward_bound = np.percentile(rewards, args.percentile)
        training_obs = []
        training_act = []
        for ep in episodes:
            if ep.reward > reward_bound:
                training_obs.extend([a_s.observation for a_s in ep.state_action_pair])
                training_act.extend([a_s.action for a_s in ep.state_action_pair])


        train_obs_v = torch.FloatTensor(training_obs)
        train_act_v = torch.LongTensor(training_act)

        optimizer.zero_grad()
        actions_logits_v = cem_net(train_obs_v)
        loss_v = ce_loss(actions_logits_v, train_act_v)
        loss_v.backward()
        optimizer.step()

        if mean_reward > 199:
            print("Solved!")
            break

    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='cem')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--percentile', type=float, default=70)

    args = parser.parse_args()

    cem(args)



