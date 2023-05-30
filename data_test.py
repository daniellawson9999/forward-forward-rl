import gymnasium as gym
import openai

from collections import deque
import argparse
import os
import pathlib
import pickle

import numpy as np

        
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
            action = policy.act(state)
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



class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return self.env.action_space.sample()

class KNNAgent:
    def __init__(self, env, actions, obs, k):
        self.env = env 
        self.actions = actions
        self.obs = obs
        self.k = k
    
    def act(self, obs):
        # compute distance to all observations
        dists = np.linalg.norm(self.obs - obs, axis=1)
        # find k nearest neighbors
        indices = np.argsort(dists)[:self.k]

        actions = self.actions[indices]

        # check if discrete action space from env.action_space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # return action with highest frequency if discrete action space
            action = np.bincount(actions).argmax()
        else:
            # compute mean action if continuous action space
            action =  np.mean(actions, axis=0)

        return action

# GPT Agent
class GPTAgent:
    def __init__(self, actions, obs, round=2, include_desc=False):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.actions = actions
        self.obs = obs
        self.round = round
        self.max_tokens = 3
        self.include_desc = include_desc
    
    def obs_actions_to_str(self, p_obs, p_actions, current_obs):
        if round is not None:
            p_obs = np.round(p_obs, self.round)
            current_obs = np.round(current_obs, self.round)
            p_actions = np.round(p_actions, self.round)
            N = len(p_obs)

            prompt = ''
            if self.include_desc:
                #desc = 'You are generating actions for a simulated robot. Here are examples of mapping obs to the correct action\n'
                desc = 'You are generating actions for a simulated robot. Here are examples of mapping obs to the correct action\n'
                prompt += desc
            for i in range(N):
                prompt += f'obs: {p_obs[i]} act: {p_actions[i]}\n'
            # if self.include_desc:
            #     prompt += 'Here is the current observation. What action would you take?\n'
            prompt += f'obs: {current_obs} act:'
            #import pdb; pdb.set_trace()
            return prompt
    
    def act(self, obs):
        prompt = self.obs_actions_to_str(self.obs, self.actions, obs)
        #import pdb; pdb.set_trace()
        response = openai.Completion.create(model=args.gpt_model, prompt=prompt, temperature=0, max_tokens=self.max_tokens, stop=['\n'])
        action = int(response.choices[0].text)
        return action

def main(args):
    current_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = os.path.join(current_dir, "data")
    data_path = os.path.join(data_dir, args.data)

    with open(data_path, 'rb') as handle:
        dataset = pickle.load(handle)
    print("mean return in dataset", dataset['rewards'].sum() / dataset['dones'].sum())


    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset['observations']), size=args.n_examples, replace=False)
    obs = dataset['observations'][indices]
    actions = dataset['actions'][indices]

    render_mode = 'human' if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    # evaluate agent
    if args.agent_type == 'random':
        agent = RandomAgent(env)
    elif args.agent_type == 'knn':
        agent = KNNAgent(env, actions, obs, k=args.knn_k)
    elif args.agent_type == 'gpt':
        print('Using', args.gpt_model)
        agent = GPTAgent(actions, obs, round=args.round, include_desc=args.include_desc)
    else:
        raise NotImplementedError
    
    mean_return, std_return, _ = eval(agent, env, args.n_eval_episodes, collect=False)
    print(f'mean return: {mean_return}\tstd: {std_return}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1', type=str)
    parser.add_argument('--render', default=False,action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--agent_type', type=str, default='random')

    parser.add_argument('--knn_k', type=int, default=1)
    parser.add_argument('--gpt_model', type=str, default='ada')
    parser.add_argument('--round', type=int, default=2)
    parser.add_argument('--include_desc', default=False, action='store_true')

    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--n_examples', type=int, default=100)
    parser.add_argument('--n_eval_episodes', type=int, default=10)

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    main(args)