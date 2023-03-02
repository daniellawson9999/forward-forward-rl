import numpy as np

from collections import deque
import argparse
import pathlib
import os
import pickle
import random

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym

# Based off: https://github.com/huggingface/deep-rl-class/blob/main/notebooks/unit4/unit4.ipynb

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size, n_layers, deterministic=False):
        super(Policy, self).__init__()
        # self.fc1 = nn.Linear(s_size, h_size)
        # self.fc2 = nn.Linear(h_size, a_size)
        layers = []
        layers.append(nn.Linear(s_size, h_size))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2): 
            layers.append(nn.Linear(h_size, h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(h_size, a_size))
        self.fc = nn.Sequential(*layers)
        self.deterministic = deterministic

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        device = self.fc[0].weight.device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        if self.deterministic:
            action = probs.argmax()
        else:
            action = m.sample()
        return action.item(), m.log_prob(action)
    
def reinforce(policy, env, optimizer, args):
    gamma = args.gamma
    print_every = args.log_interval
    n_training_episodes = args.train_episodes

    reward_threshold = env.spec.reward_threshold

    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state, info = env.reset()
        done = False
        while not done:
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # calculate the return
        returns = deque(maxlen=len(rewards)) 
        n_steps = len(rewards) 

        # Compute the discounted returns at each timestep,
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )    
            
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        # PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            if reward_threshold is not None and args.break_at_threshold and avg_score > reward_threshold:
                print("Reached threshold, stopping training...")
                break
        
    return scores
def eval(policy, env, n_eval_episodes, collect):
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    returns = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        done = False
        total_rewards_ep = 0
        
        while not done:
            action, _ = policy.act(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rewards_ep += reward

            if collect:
                dataset['observations'].append(state)
                dataset['actions'].append(action)
                dataset['rewards'].append(reward)
                dataset['dones'].append(int(done))

            state=new_state
            
        returns.append(total_rewards_ep)

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if collect:
        for key, data in dataset.items():
            dataset[key] = np.array(data)

    return mean_return, std_return, dataset




def main(args):
    render_mode = 'human' if args.render else None

    if 'ALE' in args.env:
        env = gym.make(args.env, render_mode=render_mode,  obs_type='ram')
    else:
        env = gym.make(args.env, render_mode=render_mode)

    print('action_space', env.action_space)
    print('obs_space', env.observation_space)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 
    
    policy = Policy(state_size, action_size, h_size=args.h, n_layers=args.layers, deterministic=args.deterministic_training)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Train policy
    reinforce(policy, env, optimizer, args)

    # Eval
    if args.eval_episodes > 0:
        #policy.deterministic = True
        mean_return, std_return, dataset = eval(policy, env, args.eval_episodes, collect=args.collect)
        print(f'mean return: {mean_return}\tstd: {std_return}')

        # Save trajectories
        if args.collect:
            timesteps = dataset['observations'].shape[0]
            episodes = dataset['dones'].sum()
            data_id = random.randint(int(1e4), int(1e5) - 1)
            print(f"Collect {episodes} trajectories with {timesteps} timesteps")
            current_dir = pathlib.Path(__file__).parent.resolve()
            data_dir = os.path.join(current_dir, args.save_dir)
            data_name = f'data-t_{timesteps}-i_{data_id}.pkl'
            data_path = os.path.join(data_dir, data_name)

            with open(data_path, 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        import pdb; pdb.set_trace()

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1', type=str)
    parser.add_argument('--render', default=False,action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)

    parser.add_argument('--train_episodes', type=int, default=2500)
    parser.add_argument('-dt', '--deterministic_training', default=False, action='store_true')
    parser.add_argument('--break_at_threshold', default=False, action='store_true')
    
    parser.add_argument('--eval_episodes', type=int, default=0)
    parser.add_argument('--collect', default=False, action='store_true', help='collect data during evaluation')
    parser.add_argument('--save_dir', default='data', type=str)

    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    assert(args.layers >= 2), "need at least two layers"

    main(args)