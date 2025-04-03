"""
    #! File: This file include PPO-policy's main framework.
    #?
    #*
"""
import gym 
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from gymnasium.core import Env
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

log_path = './log/tmp'
# writer = SummaryWriter(log_path)

class Policy_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):   
        """
        # @param x: x.shape = batch_size * state_dim
        # @return: 
        """
                                                        
        x = F.relu(self.fc1(x))     #* softmax's dim  = 1
                                
        return F.softmax(self.fc2(x), dim=1)    #* \frac{exp(xi)}{\sum_j \exp(xj)}

class Value_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):   
        """
        # @param x: x.shape = batch_size * state_dim
        # @return: 
        
        """
        x = F.relu(self.fc1(x))

        return self.fc2(x)

def compute_gae(gamma, lambda_, td_delta):
    """
    
    """
    td_delta = td_delta.detach().cpu().numpy() # "slice" below needs numpy().
    # print(td_delta)
    advantage_list = []
    advantage = 0
    for delta in td_delta[::-1]:    # from end to start: while delta is a [...]_dim=1
        advantage = gamma * lambda_ * advantage + delta     # \delta_0 multiplt no lambda_ and gamma.
        advantage_list.append(advantage)
    # print(advantage_list)
    advantage_list.reverse()    # change \delta_0 to advantage_list[0]
    return torch.tensor(advantage_list, dtype=torch.float)
    


class PPO_Clip:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gae_lambda, epochs, clip_eps, gamma, device):
        """
        # @param gae_lambda: Used in Generial Advance Function Evaluate.
        # @param epochs:
        # @param clip_eps: Used in PPO-Clip Algorithm.

        """
        self.actor  = Policy_Net(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device=device)
        self.critic = Value_Net(state_dim=state_dim, hidden_dim=hidden_dim).to(device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.device = device
        
    def take_action(self, state):
        """
        # ? This Function is going to choice an ACTION to interact with ENV.
        # @param state: A 1-dim list which Got from RL_ENV.
        # @return: Return The Choiced Action.

        """
        # print("state is", state)
        state = torch.tensor([state], dtype=torch.float).to(device=self.device) # Add a dim to state, which used as batch_size_dim
        probs = self.actor(state)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        # Choice The max probability action.
        return action.item()
    
    def update(self, trans_dict):
        states = torch.tensor(np.array(trans_dict['state'])).to(self.device) # Add a dim.
        # print(states.shape)
        actions = torch.tensor(trans_dict['action']).view(-1, 1).to(self.device) # Add a dim.
        # print(actions)
        rewards =torch.tensor(trans_dict['reward'], dtype=torch.float).view(-1,1).to(self.device) # Add a dim. Add a type.
        # print(rewards)
        dones = torch.tensor(trans_dict['done'],dtype=torch.float).view(-1, 1).to(self.device) # Add a dim.
        # print(dones)
        next_states = torch.tensor(np.array(trans_dict['next_state'])).to(self.device) # Add a dim.
        # print(next_states)
        # print(self.critic(states))
        td_target = rewards + self.gamma * self.critic(states)*(1-dones) # delta = r + b*v - v
        # print(td_target)
        td_delta =td_target - self.critic(states) # After critic-net forward twice, does grident compute twice ???# todo
        # print(td_delta)
        GAE = compute_gae(gamma=self.gamma, lambda_=self.gae_lambda, td_delta=td_delta).to(self.device)
        # print(GAE)
        log_old_policy_as = torch.log(self.actor(states).gather(1, actions)).detach() # find \pi_old (a | s)

        for _ in range(self.epochs):
            new_log_policy_as = torch.log(self.actor(states).gather(1, actions)) # no-detach
            # print(new_log_policy_as)
            radio = torch.exp(new_log_policy_as - log_old_policy_as) # old_policy / new_policy
            # print(radio) ALL 1 ???? 
            ppo_min_x = radio * GAE
            ppo_min_y = torch.clamp(radio, 1-self.clip_eps, 1+self.clip_eps) * GAE # how does climp compute grident
            # print(ppo_min_y)
            actor_loss = torch.mean(-torch.min(ppo_min_x, ppo_min_y)) # argmax so SGD-Ascend therefore "-"
            # print(actor_loss)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            # 
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

# ! New Version Gymnasium Support Parallel ENV RUNNING. Use 1 temp.
envs = gym.make("CartPole-v1", render_mode='human')       

_, _ = envs.reset(seed=0)

# print(envs.observation_space.shape)


state_num = envs.observation_space.shape[0]

action_num = envs.action_space.n

# print(action_num)

hidden_dim = 128

gamma = 0.98

gae_lambda = 0.95

epochs = 10

actor_lr = 1e-3

critic_lr = 1e-2

clip_eps = 0.2

episode_num = 500

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
print("Backend device is ", device)

def train_on_policy(env:Env, agent:PPO_Clip, num_episodes):
    """ 
        #* @brief: It is a general train-pipeline.
        #* @param: GYM-ENV
        #* @param: PPO-Agent
        #* @param: Episode-Number
        #* No return.
    """
    return_ls = []

    for i in tqdm(range(num_episodes)):
        trans_dict = {'state':[], 'action':[], 'next_state':[], 'reward':[], 'done':[]}
        episode_return = 0
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.take_action(state=state)
            next_state, reward, done, tranc, _ = env.step(action=action)
            if tranc == True:
                done = True
            
            trans_dict['state'].append(state)
            trans_dict['action'].append(action)
            trans_dict['reward'].append(reward)
            trans_dict['done'].append(done)
            trans_dict['next_state'].append(next_state)

            state = next_state 

            episode_return += reward
        
        # return_ls.append(episode_num) # save each episode return.

        agent.update(trans_dict=trans_dict)


if __name__ == '__main__':

    bignut = PPO_Clip(state_dim=state_num, hidden_dim=hidden_dim, action_dim=action_num,
                      actor_lr=actor_lr, critic_lr=critic_lr, clip_eps=clip_eps,
                      gae_lambda=gae_lambda, epochs=epochs, gamma=gamma, device=device)
    

    train_on_policy(env=envs, agent=bignut, num_episodes=episode_num)
    


    
