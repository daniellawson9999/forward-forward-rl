import copy 
import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


class ForwardPolicy(torch.nn.Module):

    def __init__(self, state_size, action_size, h_size,  n_layers, opt_settings, threshold, return_scale, horizon_scale, device):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.threshold = threshold
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        first_dim = state_size + action_size + 2 # Two extra dims for reward and horizon
        #first_dim = state_size + action_size + 1 # Two extra dims for reward and horizon
        dims = [first_dim] # add extra dimension for onehot action

        for _ in range(n_layers):
            dims.append(h_size)
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], threshold=threshold, opt_settings=opt_settings).to(device=self.device)]
    
    def forward(self, state, command):
        goodness_per_action = []
        for action in range(self.action_size):
            action = torch.tensor(action, device=self.device)#.unsqueeze(0)
            h = self.combine_transitions(state, action, command)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_action += [sum(goodness)]
        goodness_per_action = torch.cat(goodness_per_action)
        return goodness_per_action


    
    def act(self, state, command, deterministic=False, temperature=1):
        # state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        # target_return = torch.tensor(target_return, device=0, dtype=torch.float32).reshape(1,1)
        state = state.unsqueeze(0)
        command = command.unsqueeze(0)
        goodness_per_action = self.forward(state, command)

        if deterministic:
            action = goodness_per_action.argmax().item()
        else:
            action = torch.multinomial(F.softmax(goodness_per_action, dim=0),1).item()
        return action

    def train(self, states, actions, scores, commands):
        # states = torch.tensor(np.stack(states), device=self.device, dtype=torch.float32)
        # actions = torch.tensor(np.array(actions), device=self.device)
        states = self.combine_transitions(states, actions, commands)
        mean_loss = []
        
        h = states
        for i, layer in enumerate(self.layers):
            h,loss = layer.train(h, scores)
            mean_loss.append(loss)
        
        mean_loss = np.mean(mean_loss)
        return mean_loss

    # Wrapping w/ policy
    def combine_transitions(self, states, actions, commands):
        onehot_actions = F.one_hot(actions, num_classes=self.action_size).reshape(-1, self.action_size).to(device=self.device)
        commands[:, 0] = commands[:,0] * self.return_scale
        commands[:, 1] = commands[:,1] * self.horizon_scale
        #commands = commands[:, 0].reshape(-1,1)
        states = torch.concat([states, onehot_actions, commands], dim=-1)
        return states
    

class Layer(nn.Linear):
    def __init__(self, in_features, out_features, opt_settings, threshold=2,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = optim.Adam(self.parameters(), lr=opt_settings['lr'])
        self.threshold = threshold
        self.inner_updates = opt_settings['inner_updates']

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
        return self.forward(x).detach(), loss.detach().cpu().item()
    
class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
        
    def add_sample(self, states, actions, rewards):
        episode = {"states": states, "actions":actions, "rewards": rewards, "summed_rewards":sum(rewards)}
        self.buffer.append(episode)

        

    def sort(self):
        #sort buffer
        self.buffer = sorted(self.buffer, key = lambda i: i["summed_rewards"],reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]

    def get_random_samples(self, batch_size):
        #self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch

    def get_nbest(self, n):
        #self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)
    
# FUNCTIONS FOR Sampling exploration commands

def sampling_exploration(buffer, n_best):
    """
    This function calculates the new desired reward and new desired horizon based on the replay buffer.
    New desired horizon is calculted by the mean length of the best last X episodes. 
    New desired reward is sampled from a uniform distribution given the mean and the std calculated from the last best X performances.
    where X is the hyperparameter last_few.
    
    """
    
    top_X = buffer.get_nbest(n_best)
    #The exploratory desired horizon dh0 is set to the mean of the lengths of the selected episodes
    new_desired_horizon = np.mean([len(i["states"]) for i in top_X])
    # save all top_X cumulative returns in a list 
    returns = [i["summed_rewards"] for i in top_X]
    # from these returns calc the mean and std
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)
    # sample desired reward from a uniform distribution given the mean and the std
    new_desired_reward = np.random.uniform(mean_returns, mean_returns+std_returns)

    return torch.FloatTensor([new_desired_reward])  , torch.FloatTensor([new_desired_horizon]) 


def select_time_steps(saved_episode):
    """
    Given a saved episode from the replay buffer this function samples random time steps (t1 and t2) in that episode:
    T = max time horizon in that episode
    Returns t1, t2 and T 
    """
    # Select times in the episode:
    T = len(saved_episode["states"]) # episode max horizon 
    t1 = np.random.randint(0,T-1)
    t2 = np.random.randint(t1+1,T)

    return t1, t2, T

def create_training_input(episode, t1, t2):
    """
    Based on the selected episode and the given time steps this function returns 4 values:
    1. state at t1
    2. the desired reward: sum over all rewards from t1 to t2
    3. the time horizont: t2 -t1
    
    4. the target action taken at t1
    
    buffer episodes are build like [cumulative episode reward, states, actions, rewards]
    """
    state = episode["states"][t1] 
    desired_reward = sum(episode["rewards"][t1:t2])
    time_horizont = t2-t1
    action = episode["actions"][t1]
    return state, desired_reward, time_horizont, action

def create_training_examples(buffer, batch_size, args):
    """
    Creates a data set of training examples that can be used to create a data loader for training.
    ============================================================
    1. for the given batch_size episode idx are randomly selected
    2. based on these episodes t1 and t2 are samples for each selected episode 
    3. for the selected episode and sampled t1 and t2 trainings values are gathered
    ______________________________________________________________
    Output are two numpy arrays in the length of batch size:
    Input Array for the Behavior function - consisting of (state, desired_reward, time_horizon)
    Output Array with the taken actions 
    """
    # input_array = []
    # output_array = []
    states, actions, commands = [], [], []

    # select randomly episodes from the buffer
    episodes = buffer.get_random_samples(batch_size // 2)
    for ep in episodes:
        #select time stamps
        t1, t2, T = select_time_steps(ep)
        # For episodic tasks they set t2 to T:
        t2 = T
        state, desired_reward, time_horizont, action = create_training_input(ep, t1, t2)
        states.append(state)
        actions.append(action)
        commands.append(torch.cat([torch.FloatTensor([desired_reward]), torch.FloatTensor([time_horizont])]).to(device=args.device))
        #input_array.append(torch.cat([state, torch.FloatTensor([desired_reward]), torch.FloatTensor([time_horizont])]))
        #output_array.append(action)

    states = torch.stack(states).to(device=args.device)
    actions = torch.tensor(actions).reshape(-1,1).to(device=args.device)
    commands = torch.stack(commands)
    scores = torch.ones_like(actions, device=args.device).reshape(-1,1)

    # add negative examples
    states = torch.cat([states, states])
    commands = torch.cat([commands, commands])
    # TODO, update sample when action dim > 2
    actions = torch.cat([actions, 1 - actions])
    scores = torch.cat([scores, -scores])

    return states, actions, commands, scores

def training_update(policy, buffer, batch_size, args):
    """
    Trains the BF with on a cross entropy loss were the inputs are the action probabilities based on the state and command.
    The targets are the actions appropriate to the states from the replay buffer.
    """
    states, actions, commands, scores = create_training_examples(buffer, batch_size, args)
    loss = policy.train(states, actions, scores, commands)

    return loss

def evaluate_policy(env, policy, desired_return, desired_time_horizon):
    """
    Runs one episode of the environment to evaluate the bf.
    """
    state,_ = env.reset()
    rewards = 0
    while True:
        state = torch.FloatTensor(state)
        action = policy.act(state.to(policy.device), torch.concat([desired_return, desired_time_horizon]).to(policy.device))
        #state, reward, done, _ = env.step(action.cpu().numpy())
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated 
        rewards += reward
        desired_return = min(desired_return - reward, torch.FloatTensor([args.max_return]))
        desired_time_horizon = max(desired_time_horizon - 1, torch.FloatTensor([1]))
        
        if done:
            break 
    return rewards

# Algorithm 2 - Generates an Episode unsing the Behavior Function:
def generate_episode(env, policy, desired_return, desired_time_horizon):    
    """
    Generates more samples for the replay buffer.
    """
    state,_ = env.reset()
    states = []
    actions = []
    rewards = []
    while True:
        state = torch.FloatTensor(state)

        action = policy.act(state.to(policy.device), torch.concat([desired_return, desired_time_horizon]).to(policy.device))
        #next_state, reward, done, _ = env.step(action.cpu().numpy())
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated 
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        desired_return -= reward
        desired_time_horizon -= 1
        desired_time_horizon = torch.FloatTensor([np.maximum(desired_time_horizon, 1).item()])
        
        if done:
            break 
    return [states, actions, rewards]


# Algorithm 1 - Upside - Down Reinforcement Learning 
def run_upside_down(env, policy, buffer, iter, args):
    """
    
    """
    all_rewards = []
    losses = []
    average_100_reward = []
    desired_rewards_history = []
    horizon_history = []
    for ep in range(1, iter+1):

        # improve|optimize bf based on replay buffer
        loss_buffer = []
        for i in range(args.updates_per_iter):
            loss = training_update(policy, buffer, args.batch_size, args)
            loss_buffer.append(loss)
        policy_loss = np.mean(loss_buffer)
        losses.append(policy_loss)
        
        # run x new episode and add to buffer
        for i in range(args.episodes_per_iter):
            
            # Sample exploratory commands based on buffer
            new_desired_reward, new_desired_horizon = sampling_exploration(buffer, args.top_n)
            generated_episode = generate_episode(env, policy, new_desired_reward, new_desired_horizon)
            buffer.add_sample(generated_episode[0],generated_episode[1],generated_episode[2])
        buffer.sort()
            
        new_desired_reward, new_desired_horizon = sampling_exploration(buffer, args.top_n)
        # monitoring desired reward and desired horizon
        desired_rewards_history.append(new_desired_reward.item())
        horizon_history.append(new_desired_horizon.item())
        
        ep_rewards = evaluate_policy(env, policy, new_desired_reward, new_desired_horizon)
        all_rewards.append(ep_rewards)
        average_100_reward.append(np.mean(all_rewards[-100:]))
        


        print("Episode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.4f}".format(ep, ep_rewards, np.mean(all_rewards[-100:]), policy_loss))
        if ep % 100 == 0:
            print("Episode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.4f}".format(ep, ep_rewards, np.mean(all_rewards[-100:]), policy_loss))
            
    return all_rewards, average_100_reward, desired_rewards_history, horizon_history, losses

def train_policy(env, policy, args):
    buffer = ReplayBuffer(args.replay_size)
    
    # Collect data during n-warmup rounds
    init_desired_return = 1
    init_time_horizon = 1

    for i in range(args.warmup_episodes):
        desired_return = torch.FloatTensor([init_desired_return])
        desired_time_horizon = torch.FloatTensor([init_time_horizon])
        state,_ = env.reset()
        states = []
        actions = []
        rewards = []
        while True:
            action = policy.act(torch.from_numpy(state).float().to(args.device), torch.concat([desired_return, desired_time_horizon]).to(policy.device))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(torch.from_numpy(state).float())
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            desired_return -= reward
            desired_time_horizon -= 1
            desired_time_horizon = torch.FloatTensor([np.maximum(desired_time_horizon, 1).item()])

            if done:
                break 
            
        buffer.add_sample(states, actions, rewards)
    buffer.sort()
    
    # Run main upside-down training loop
    rewards, average, desired_rewards, desired_horizon, loss = run_upside_down(env, policy, buffer, args.iter, args)

    if args.plot:
        plot_results(rewards, average, desired_rewards, desired_horizon, loss)

def plot_results(rewards, average, desired_rewards, desired_horizon, loss):
    plt.figure(figsize=(15,8))
    plt.subplot(2,2,1)
    plt.title("Rewards")
    plt.plot(rewards, label="rewards")
    plt.plot(average, label="average100")
    plt.legend()
    plt.subplot(2,2,2)
    plt.title("Loss")
    plt.plot(loss)
    plt.subplot(2,2,3)
    plt.title("desired Rewards")
    plt.plot(desired_rewards)
    plt.subplot(2,2,4)
    plt.title("desired Horizon")
    plt.plot(desired_horizon)
    plt.show()

def main(args):
    render_mode = 'human' if args.render else None

    env = gym.make(args.env, render_mode=render_mode)
    
    print('action_space', env.action_space)
    print('obs_space', env.observation_space)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Update?
    args.desired_return = 200
    args.max_return = args.desired_return
    args.desired_horizon = 200
    # args.horizon_scale = 1
    # args.return_scale =  1
    args.horizon_scale = args.desired_horizon / 10000 # 0.02
    args.return_scale = args.desired_return / 10000


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
        return_scale=args.return_scale,
        horizon_scale=args.horizon_scale
    )

    train_policy(env, policy, args)

    evaluate_policy(env, policy, 
        desired_return = torch.FloatTensor([args.desired_return]).to(device=args.device), 
        desired_time_horizon = torch.FloatTensor([args.desired_horizon]).to(device=args.device)
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1', type=str)
    parser.add_argument('--render', default=False,action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--plot', default=False, action='store_true')

    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--inner_updates', default=10, type=int)
    parser.add_argument('--threshold', type=float, default=2)
    parser.add_argument('--batch_size', type=int, default=256)


    parser.add_argument('--replay_size', type=int, default=700)
    parser.add_argument('--warmup_episodes', default=50, type=int)
    parser.add_argument('--iter', type=int, default=200)
    parser.add_argument('--updates_per_iter', type=int, default=100)
    parser.add_argument('--episodes_per_iter', type=int, default=15)
    parser.add_argument('--top_n', type=int, default=15)

    #parser.add_argument('-dt', '--deterministic_training', default=False, action='store_true')
    parser.add_argument('--break_at_threshold', default=False, action='store_true')



    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    assert(args.layers >= 2), "need at least two layers"

    main(args)