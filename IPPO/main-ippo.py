"""
    #! File: 
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
import highway_env


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
        state = state.reshape(-1) # 5 * 7 state_input reshape to [1,]
        state = torch.tensor([state], dtype=torch.float).to(device=self.device) # Add a dim to state, which used as batch_size_dim
        # print(" state is", state, "shape is ", state.shape)
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

        trans_dict['state'] = np.array(trans_dict['state'])
        trans_dict['state'] = trans_dict['state'].reshape((trans_dict['state'].shape[0], -1))

        trans_dict['next_state'] = np.array(trans_dict['next_state'])
        trans_dict['next_state'] = trans_dict['next_state'].reshape((trans_dict['next_state'].shape[0], -1))

        # print(trans_dict['state'].shape)
        # print(.reshape(trans_dict['state'].shape[0], -1));return
        states = torch.tensor(trans_dict['state']).to(self.device) # Add a dim.
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


controlled_num = 5
uncontrolled_num = 20

env = gym.make('highway-v0', 
    render_mode='rgb_array',
    config={
    "controlled_vehicles": controlled_num,
    "vehicles_count": uncontrolled_num,
    "ego_spacing" : 0.5,
    "lanes_count": 12,
    "vehicles_density": 0.8,
    "simulation_frequency": 20,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "normalize_reward": True,
    "collision_reward": -1, # 超参
    "lane_change_reward": -0.05, # 变道惩罚
    "reward_speed_range": [20, 30],  # 实际车速范围
    "initial_lane_id": None,
    "screen_width": 1800,  # [px]
    "screen_height": 600,  # [px]
    "centering_position": [0.1, 0.35],
    "scaling": 5,
    "action": {
        "type": "MultiAgentAction",

        "action_config": {
            "type": "DiscreteMetaAction",
        }
    },
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            'lanes_count': 6,
            "vehicles_count": 5, # 扩大观测空间
            "features": ["presence", "x", "y", "vx", "vy","cos_h","sin_h"], # 对应修改，说不定presence可以去掉，或者说把vehicles_count参数调大
            "features_range": {
                "x": [-200, 200], # 防止归一化之后全是1 
                "y": [-12, 12],
                "vx": [-80, 80], 
                "vy": [-80, 80]
            },
            "absolute": False, # 绝对位置,如果最后还是训练不出来，就改成相对的
            "normalize" : True, # obs归一化
            "order": "sorted", # "order"，"shuffled"
        }   
    }
})


obs, _ = env.reset()
env.render()

state_num = obs[0].shape[0] * obs[0].shape[1]

action_num = env.action_space[0].n

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


def train_on_policy(env:Env, agent:PPO_Clip, num_episodes):
    """ 
        #* @brief: It is a general train-pipeline.
        #* @param: GYM-ENV
        #* @param: PPO-Agent
        #* @param: Episode-Number
        #* No return.
    """
    # return_ls = []

    for i in tqdm(range(num_episodes)):
        trans_dict_ls = []
        for _ in range(controlled_num):
            tmp_trans_dict= {'state':[], 'action':[], 'next_state':[], 'reward':[], 'done':[]}
            trans_dict_ls.append(tmp_trans_dict)

        # print(trans_dict_ls)
        state, _ = env.reset()
        env.render()

        done = False
        # crash = False
        episode_return = 0

        # state_ls = [0] * controlled_num
        action_ls = [0] * controlled_num
        # next_state_ls = [0] * controlled_num

        state_ls = list(state)

        while not done:
            for i in range(controlled_num):
                action_ls[i] = agent.take_action(state=state_ls[i])
            
            next_state, reward, done, tranc, info = env.step(action=tuple(action_ls))
            # print(reward)
            # crash = info['crashed']
            env.render()
            
            next_state_ls = list(next_state)

            if tranc == True:
                done = True
            
            for i in range(controlled_num):
                
                # print(trans_dict_ls[i])
                trans_dict_ls[i]['state'].append(state_ls[i])
                trans_dict_ls[i]['action'].append(action_ls[i])
                trans_dict_ls[i]['reward'].append(reward[i])
                trans_dict_ls[i]['done'].append(done)
                trans_dict_ls[i]['next_state'].append(next_state_ls[i])

            for i in range(controlled_num):
                state_ls[i] = next_state_ls[i]

            episode_return += sum(reward)
        
        # return_ls.append(episode_num) # save each episode return.

        #! Update until All Done.
        for i in range(controlled_num):
            agent.update(trans_dict=trans_dict_ls[i])
            # agent.update(trans_dict=trans_dict_0)
            # agent.update(trans_dict=trans_dict_1)
        # break


if __name__ == '__main__':

    bignut = PPO_Clip(state_dim=state_num, hidden_dim=hidden_dim, action_dim=action_num,
                      actor_lr=actor_lr, critic_lr=critic_lr, clip_eps=clip_eps,
                      gae_lambda=gae_lambda, epochs=epochs, gamma=gamma, device=device)
    

    train_on_policy(env=env, agent=bignut, num_episodes=episode_num)
    


    