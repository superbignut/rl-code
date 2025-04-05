"""
    #! File: Combat env is so hard to understand.
    #?
    #*
"""
# import ma_gym
# import gym 
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from gymnasium.core import Env
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ma_gym.envs.combat.combat import Combat

log_path = './log/ippo'
# writer = SummaryWriter(log_path)

class Policy_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim) # Add More.
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):   
        """
        # @param x: x.shape = batch_size * state_dim
        # @return: 
        """
                                                        
        x = F.relu(self.fc1(x))     #* softmax's dim  = 1
        x = F.relu(self.fc2(x)) 
        return F.softmax(self.fc3(x), dim=1)    #* \frac{exp(xi)}{\sum_j \exp(xj)}

class Value_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim) # Add More.
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):   
        """
        # @param x: x.shape = batch_size * state_dim
        # @return: 
        
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        """
            # ? This Function is going to Update Policy_Net and Value_Net When Each-Episode Ending.
            # @param trans_dict: trans_dict include s, a, r, s', done. Each Value of this dict is the Date Collected 
                                 from One-Episode Simulation. #! Attention. One-Episode Simulation.
                                 Therefore, eg. states.shape is like (k, state_dim), reward's shape is (k, 1)
            # @return: No Return.
        
        """
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
        td_target = rewards + self.gamma * self.critic(next_states)*(1 - dones) # delta = r + b*v' - v
        # print(td_target)
        td_delta = td_target - self.critic(states) # 
        # print(td_delta)
        GAE = compute_gae(gamma=self.gamma, lambda_=self.gae_lambda, td_delta=td_delta).to(self.device) # GAE has no CG.
        # print(GAE)
        log_old_policy_as = torch.log(self.actor(states).gather(1, actions)).detach() # find \pi_old (a | s) with no CG.

        for _ in range(self.epochs):
            """
                @* This loop is going to iterate new_policy / old_policy in ppo, althought we don't konw new_policy actually.
                   But We can random choose a new_policy_0 and iterate epochs times. After each optimize, PPO ensure to maxium 
                   the Return(State_Value_Function) and Policy_Obj_Function.
                #! Attention. 
                   All variables/tensors unrelated to Policy_Theta and Value_Output need to be detached.
                #* Such as: P_GAE, P_old_pi_as in Policy-Net and td_target in Value-Net.

            """
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
            # optimize.

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()



grid_size = (15, 15)
team_size = 2

envs = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size) # 

envs.reset()

# print(envs.observation_space.shape)


state_num = envs.observation_space[0].shape[0]
print(state_num)

action_num = envs.action_space[0].n

print(action_num)

hidden_dim = 128

gamma = 0.98

gae_lambda = 0.95

epochs = 10

actor_lr = 3e-4

critic_lr = 1e-3

clip_eps = 0.2

episode_num = 500

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
print("Backend device is ", device)



win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transition_dict_1 = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            transition_dict_2 = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            s = env.reset()
            terminal = False
            while not terminal:
                a_1 = agent.take_action(s[0])
                a_2 = agent.take_action(s[1])
                next_s, r, done, info = env.step([a_1, a_2])
                transition_dict_1['states'].append(s[0])
                transition_dict_1['actions'].append(a_1)
                transition_dict_1['next_states'].append(next_s[0])
                transition_dict_1['rewards'].append(
                    r[0] + 100 if info['win'] else r[0] - 0.1)
                transition_dict_1['dones'].append(False)
                transition_dict_2['states'].append(s[1])
                transition_dict_2['actions'].append(a_2)
                transition_dict_2['next_states'].append(next_s[1])
                transition_dict_2['rewards'].append(
                    r[1] + 100 if info['win'] else r[1] - 0.1)
                transition_dict_2['dones'].append(False)
                s = next_s
                terminal = all(done)
            win_list.append(1 if info["win"] else 0)
            agent.update(transition_dict_1)
            agent.update(transition_dict_2)
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)
