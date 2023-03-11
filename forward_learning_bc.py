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



scale = 500
target_return = 500 / scale
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class ForwardPolicy(torch.nn.Module):

    def __init__(self, state_size, action_size, h_size,  n_layers, opt_settings, threshold, device, deterministic, model_reward):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.threshold = threshold
        self.deterministic = deterministic
        self.model_reward = model_reward

        first_dim = state_size + action_size
        if model_reward:
            first_dim += 1
        dims = [first_dim] # add extra dimension for onehot action

        for _ in range(n_layers):
            dims.append(h_size)
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], threshold=threshold, opt_settings=opt_settings).to(device=self.device)]
    
    def act(self, state, target_return):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        target_return = torch.tensor(target_return, device=0, dtype=torch.float32).reshape(1,1)
        goodness_per_action = []
        for action in range(self.action_size):
            action = torch.tensor(action, device=self.device).unsqueeze(0)
            h = self.combine_transitions(state, action, target_return)
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

    def train(self, states, actions, scores, returns):
        states = torch.tensor(np.stack(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=self.device)
        states = self.combine_transitions(states, actions, returns)
        
        h = states
        for i, layer in enumerate(self.layers):
            #print('training layer', i, '...')
            h = layer.train(h, scores)

    # Wrapping w/ policy
    def combine_transitions(self, states, actions, returns):
        onehot_actions = F.one_hot(actions, num_classes=self.action_size).to(device=self.device)
        states = torch.concat([states, onehot_actions], dim=-1)
        if self.model_reward:
            states = torch.concat([states, returns], dim=-1)
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
            h = self.forward(x)
            logits = h.pow(2).mean(1) - self.threshold
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
        rtg = target_return
        while not done:
            action, prob = policy.act(state, rtg)
            state, reward, terminated, truncated, info = env.step(action)
            rtg -= reward / scale
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
        returns = torch.tensor(returns, device=args.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        #policy.train(states, actions, returns)
        # BC, train w/ static dataset
        sampled_states,sampled_actions,sampled_scores, sampled_returns = get_batch(dataset, args.batch_size, args)
        sampled_scores = torch.tensor(sampled_scores, device=policy.device,dtype=torch.float32)
        sampled_returns = torch.tensor(sampled_returns, device=policy.device, dtype=torch.float32).reshape(-1, 1)
        policy.train(sampled_states, sampled_actions, sampled_scores, sampled_returns)
                
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            if reward_threshold is not None and args.break_at_threshold and avg_score > reward_threshold:
                print("Reached threshold, stopping training...")
                break
        
    return scores

def get_batch(dataset, batch_size, args):
    n_trajectories = dataset['observations'].shape[0]
    batch_inds = np.random.choice(
        np.arange(n_trajectories),
        size=batch_size//2,
        replace=True,
    )
    p_states = dataset['observations'][batch_inds]
    p_actions = dataset['actions'][batch_inds]
    p_returns = dataset['rtg'][batch_inds] / scale
    p_scores = np.ones_like(p_actions)

    #n_states = states
    n_returns = np.random.uniform(0,1, p_states.shape[0])
    n_actions = 1 - p_actions
    n_scores = -p_scores

    # Only actions negative
    states = np.concatenate([p_states,p_states])
    returns = np.concatenate([p_returns,p_returns])
    actions = np.concatenate([p_actions, n_actions])
    scores = np.concatenate([p_scores, n_scores])

    if args.model_reward:
        # Negative return and actions
        states = np.concatenate([states,p_states])
        returns = np.concatenate([returns,n_returns])
        actions = np.concatenate([actions, n_actions])
        scores = np.concatenate([scores, n_scores])

        # Just negative returns
        states = np.concatenate([states,p_states])
        returns = np.concatenate([returns,n_returns])
        actions = np.concatenate([actions, p_actions])
        scores = np.concatenate([scores, n_scores])
    
    return states,actions,scores,returns


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

    # Add return-to-go terms
    dataset['rtg'] = []
    ends = np.nonzero(dataset['dones'])[0]
    for i in range(len(ends)):
        if i == 0:
            start_index = 0
        else:
            start_index = ends[i - 1] + 1
        end_index = ends[i]
        ep_rtg = discount_cumsum(dataset['rewards'][start_index:end_index+1], args.gamma)
        dataset['rtg'] = np.append(dataset['rtg'], ep_rtg)

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
        model_reward=args.model_reward ,
    )


    train_policy(policy, env, dataset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1', type=str)
    parser.add_argument('--render', default=False,action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--inner_updates', default=10, type=int)
    parser.add_argument('--threshold', type=float, default=2)
    parser.add_argument('--regularize', default=False, action='store_true')

    parser.add_argument('--train_episodes', type=int, default=2500)
    parser.add_argument('-dt', '--deterministic_training', default=False, action='store_true')
    parser.add_argument('--break_at_threshold', default=False, action='store_true')

    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_reward', default=False, action='store_true')


    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    assert(args.layers >= 2), "need at least two layers"

    main(args)