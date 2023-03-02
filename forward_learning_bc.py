import numpy as np

from collections import deque
import argparse
import os
import pathlib
import pickle

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gymnasium as gym

# ForwardFordward implementation based off https://github.com/mohammadpz/pytorch_forward_forward for MNIST


class ForwardPolicy(torch.nn.Module):

    def __init__(self, state_size, action_size, h_size,  n_layers, opt_settings, threshold, device, deterministic):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.threshold = threshold
        self.deterministic = deterministic
        dims = [state_size + action_size] # add extra dimension for onehot action
        for _ in range(n_layers):
            dims.append(h_size)
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], threshold=threshold, opt_settings=opt_settings).to(device=self.device)]
    
    def act(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        goodness_per_action = []
        for action in range(self.action_size):
            action = torch.tensor(action, device=self.device).unsqueeze(0)
            h = self.combine_state_action(state, action)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_action += [sum(goodness)]
        goodness_per_action = torch.cat(goodness_per_action)

        #print("goodness", goodness_per_action)
        #print("sm",F.softmax(goodness_per_action))
        #action = goodness_per_action.argmax().item()
        action = torch.multinomial(F.softmax(goodness_per_action),1).item()
        return action, None

    def train(self, states, actions, scores):
        states = torch.tensor(np.stack(states), device=self.device)
        actions = torch.tensor(np.array(actions), device=self.device)
        states = self.combine_state_action(states, actions)
        
        h = states
        for i, layer in enumerate(self.layers):
            #print('training layer', i, '...')
            h = layer.train(h, scores)

    # Wrapping w/ policy
    def combine_state_action(self, states, actions):
        onehot_actions = F.one_hot(actions, num_classes=self.action_size).to(device=self.device)
        states = torch.concat([states, onehot_actions], dim=-1)
        return states
    

class Layer(nn.Linear):
    def __init__(self, in_features, out_features, opt_settings, threshold=2,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = optim.Adam(self.parameters(), lr=opt_settings['lr'])
        self.threshold = threshold
        self.inner_updates = opt_settings['inner_updates']
        #self.num_epochs = 1000

    def forward(self, x):
        eps = np.finfo(np.float32).eps.item()
        x_direction = x / (x.norm(2, 1, keepdim=True) + eps)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x, score):
        for i in range(self.inner_updates):
            logits = self.forward(x).pow(2).mean(1) - self.threshold
            logits = -logits * score
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(logits)).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x).detach()



    
def train_policy(policy, env, dataset, args):
    gamma = args.gamma
    print_every = args.log_interval
    n_training_episodes = args.train_episodes

    reward_threshold = env.spec.reward_threshold

    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes+1):
        actions = []
        saved_probs = []
        rewards = []
        states = []
        state, info = env.reset()
        done = False
        while not done:
            action, prob = policy.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            actions.append(action)
            saved_probs.append(prob)
            rewards.append(reward)
            states.append(state)

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
        returns = torch.tensor(returns, device=args.device)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        #policy.train(states, actions, returns)
        # BC, train w/ static dataset
        sampled_states,sampled_actions,sampled_scores = get_batch(dataset, args.batch_size)
        sampled_scores = torch.tensor(sampled_scores, device=policy.device)
        policy.train(sampled_states, sampled_actions, sampled_scores)
                
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            if reward_threshold is not None and args.break_at_threshold and avg_score > reward_threshold:
                print("Reached threshold, stopping training...")
                break
        
    return scores

def get_batch(dataset, batch_size):
    n_trajectories = dataset['observations'].shape[0]
    batch_inds = np.random.choice(
        np.arange(n_trajectories),
        size=batch_size//2,
        replace=True,
    )
    states = dataset['observations'][batch_inds]
    actions = dataset['actions'][batch_inds]
    scores = np.ones_like(actions)

    # add negative examples
    states = np.concatenate([states,states])
    actions = np.concatenate([actions, 1 - actions])
    scores = np.concatenate([scores, -scores])

    return states,actions,scores


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

    current_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = os.path.join(current_dir, "data")
    data_path = os.path.join(data_dir, args.data)
    
    with open(data_path, 'rb') as handle:
        dataset = pickle.load(handle)


    # Optimization settings to pass to each layer
    opt_settings = {
        'lr': args.lr,
        'inner_updates': args.inner_updates
    }
    
    policy = ForwardPolicy(
        state_size, action_size, 
        h_size=args.h, 
        n_layers=args.layers, 
        opt_settings=opt_settings,
        device=args.device,
        threshold=args.threshold,
        deterministic=args.deterministic_training, 
    )


    train_policy(policy, env, dataset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1', type=str)
    parser.add_argument('--render', default=False,action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')

    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--inner_updates', default=10, type=int)
    parser.add_argument('--threshold', type=float, default=2)

    parser.add_argument('--train_episodes', type=int, default=2500)
    parser.add_argument('-dt', '--deterministic_training', default=False, action='store_true')
    parser.add_argument('--break_at_threshold', default=False, action='store_true')

    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=64)


    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    assert(args.layers >= 2), "need at least two layers"

    main(args)