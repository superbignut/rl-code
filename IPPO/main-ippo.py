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
from utils import delete_dir_file # 删除文件夹下的所有文件和子文件夹
from copy import deepcopy
import subprocess
import time
from threading import Thread
# from multiprocess.

np.set_printoptions(precision=2, suppress=True) # np.array 输出打印两位小数, 禁止使用科学计数法

log_path = './log/ippo'

delete_dir_file("./log") # delete log

writer = SummaryWriter(log_path) # create log

# global_index_i = 0 # 全局索引， 勿用

class Policy_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim) # Add More.
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
    
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
        # 这里可以引入一个 max episode len
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
    def __init__(self, state_dim, hidden_dim, action_dim, ac_lr, gae_lambda, epochs, clip_eps, gamma, max_grad_norm, entropy_cof, critic_cof, device):
        """
        # @param gae_lambda: Used in Generial Advance Function Evaluate.
        # @param epochs:
        # @param clip_eps: Used in PPO-Clip Algorithm.

        """
        self.actor  = Policy_Net(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device=device)
        self.critic = Value_Net(state_dim=state_dim, hidden_dim=hidden_dim).to(device=device)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.a_c_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        
        self.a_c_optimizer = torch.optim.Adam(self.a_c_parameters, lr=ac_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.entropy_cof = entropy_cof
        self.critic_cof = critic_cof
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
            # @return: Return actor_loss_average, critic_loss_average
        
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

        actor_loss_ls = []
        critic_loss_ls = []

        for _ in range(self.epochs):
            """
                @* This loop is going to iterate new_policy / old_policy in ppo, althought we don't konw new_policy actually.
                   But We can random choose a new_policy_0 and iterate epochs times. After each optimize, PPO ensure to maxium 
                   the Return(State_Value_Function) and Policy_Obj_Function.
                #! Attention. 
                   All variables/tensors unrelated to Policy_Theta and Value_Output need to be detached.
                #* Such as: P_GAE, P_old_pi_as in Policy-Net and td_target in Value-Net.

            """
            action_probs = self.actor(states)

            action_dist = torch.distributions.Categorical(probs=action_probs)

            log_old_policy_as = torch.log(action_probs.gather(1, actions)).detach() # find \pi_old (a | s) with no CG.

            new_log_policy_as = torch.log(self.actor(states).gather(1, actions)) # no-detach
            # print(new_log_policy_as)
            radio = torch.exp(new_log_policy_as - log_old_policy_as) # old_policy / new_policy
            
            # print(radio) ALL 1 ?
            ppo_min_x = radio * GAE
            ppo_min_y = torch.clamp(radio, 1-self.clip_eps, 1+self.clip_eps) * GAE # how does climp compute grident

            # entropy 这里是在鼓励 动作分化？

            entropy_loss = action_dist.entropy() # - \sigma log(p_i) * pi

            # print(ppo_min_y)
            actor_loss = torch.mean(-torch.min(ppo_min_x, ppo_min_y) + self.entropy_cof * entropy_loss) # argmax so SGD-Ascend therefore "-"
            # print(actor_loss)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            # optimize.
            loss = actor_loss + self.critic_cof * critic_loss

            self.a_c_optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=self.a_c_parameters, max_norm=self.max_grad_norm)    

            self.a_c_optimizer.step()

        
            """ self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=self.max_grad_norm)    
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.max_grad_norm)          

            self.actor_optimizer.step()
            self.critic_optimizer.step() """

            actor_loss_ls.append(actor_loss ** 2)
            critic_loss_ls.append(critic_loss)
        
        return sum(actor_loss_ls) / len(actor_loss_ls), sum(critic_loss_ls) / len(critic_loss_ls)

controlled_num = 2
uncontrolled_num = 40


"""
simulation_frequency is the frequency used for the Euler integration of the dynamics, i.e. the physical simulation. 
A high simulation frequency will give more accurate results, at the price of an increased computational load (since
more intermediate timesteps need to be computed). I haven't experimented recently, but I think that the simulation 
tends to become unstable under 5Hz. There is no upper bound, but at some point the benefits in terms of accuracy 
become negligible. I'd say [5Hz, 15Hz] is a good range.

policy_frequency, on the other hand, is the frequency at which the agent can take decisions. It cannot be higher than
the simulation frequency, and this upper bound corresponds to the standard case where the agent provides a control 
input for every simulated frame, which makes sense for e.g. low-level control (steering/throttle). But for high-level
decisions, such as lane changes, it can be more sensible to have temporally extended actions: if a lane change takes 
at least 1 second to execute, we do not need to take decisions at a higher frequency. Increasing the policy frequency 
means that a call to the step() method will be faster, but in turn the corresponding simulated duration will be shorter.
"""
env = gym.make('highway-v0', 
    render_mode='rgb_array',
    config={
    "controlled_vehicles": controlled_num,
    "vehicles_count": uncontrolled_num,
    "ego_spacing" : 0.5,
    "lanes_count": 4,
    "vehicles_density": 0.8,
    "duration": 150,                # [s]
    "simulation_frequency": 20,     # [Hz]
    "policy_frequency": 1,          # [Hz]

    "normalize_reward": False,      # 这里似乎是 False更合理

    "collision_reward": -1,         # 碰撞奖励不易太大，否则车辆会陷入保守行驶策略

    "lane_change_reward": 0,    # 变道惩罚, 这个应该是没有使用 obsolote
    "right_lane_reward": 0.2,
    "high_speed_reward": 0.4,

    "reward_speed_range": [20, 30],  # 实际车速范围 MAX_SPEED 在kinematics 有限制 是[-40, 40]

    "initial_lane_id": None,
    "screen_width": 1200,  # [px]
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
            'lanes_count': 4,
            "vehicles_count": 6, # 扩大观测空间
            "features": ["presence", "x", "y", "vx", "vy"], # 对应修改，说不定presence可以去掉，或者说把vehicles_count参数调大
            "features_range": {
                "x": [-200, 200], # 防止归一化之后全是1 
                "y": [-12, 12],
                "vx": [-50, 50], 
                "vy": [-50, 50]
            },
            "absolute": False, # 
            "normalize": True, # obs归一化
            "order": "sorted", # "order"，"shuffled"
            "see_behind": False,
        }   
    }
})


obs, _ = env.reset()
env.render()

state_num = obs[0].shape[0] * obs[0].shape[1]

action_num = env.action_space[0].n

print("action num is : ", action_num)

hidden_dim = 512

gamma = 0.9

gae_lambda = 0.9

epochs = 20 # 

ac_lr = 6e-5     # 调的再小一点呢

# critic_lr = 1e-6    # 状态空间较大， 使用小的值函数更新速率进行迭代

clip_eps = 0.2

max_grad_norm = 0.5 # 具体的大小要根据 实际调整

entropy_cof = -0.1

critic_cof = 0.8

episode_num = 2000

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

    for episode_i in tqdm(range(num_episodes)):

        trans_dict_ls = []

        for _ in range(controlled_num):
            tmp_trans_dict= {'state':[], 'action':[], 'next_state':[], 'reward':[], 'done':[]}
            trans_dict_ls.append(tmp_trans_dict)

        # print(trans_dict_ls)
        state, _ = env.reset()
        env.render()

        done = False

        episode_return = 0

        state_ls = list(state)
        
        last_done = [False] * controlled_num # 上一个时刻就已经结束了

        all_done = False

        while not all_done:
            # print(state_ls[0])
            """ for car_i in range(controlled_num):
                action_ls[car_i] = agent.take_action(state=state_ls[car_i]) """

            action_ls = [agent.take_action(state=state_ls[car_i]) for car_i in range(controlled_num)]
            
            # print(" action is : ", action_ls)
            next_state, reward, done, tranc, info = env.step(action=tuple(action_ls))
            # print(env.get_available_actions())
            # print(" reward is ",reward)

            """
                如果一个车撞了之后, 这个reward一直是负的奖励,状态不变, 因此不应该持续更新
                代码实现见 last_done
            """
            # crash = info['crashed']
            env.render()
            
            next_state_ls = list(next_state)

            if tranc == True or all(done) == True:
                all_done = True
            
            for dict_i in range(controlled_num):
                
                if last_done[dict_i] == True:
                    # 如果上一个就已经结束了，不更新用于训练的 dict 与 episode 的 奖励输出
                    continue

                trans_dict_ls[dict_i]['state'].append(state_ls[dict_i])
                trans_dict_ls[dict_i]['action'].append(action_ls[dict_i])
                trans_dict_ls[dict_i]['reward'].append(reward[dict_i])
                trans_dict_ls[dict_i]['done'].append(done[dict_i])
                trans_dict_ls[dict_i]['next_state'].append(next_state_ls[dict_i])

                episode_return += reward[dict_i] # return更新
            
            state_ls = next_state_ls # 更新状态
            
            last_done = done # 更新 done
        
        # return_ls.append(episode_num) # save each episode return.
        # writer.add_scalar("ten_predict_acc: ", sum(return_ls[-10:]), i)

        #! Update until All Done.
        actor_loss = 0
        critic_loss = 0
        for update_i in range(controlled_num):
            _loss = agent.update(trans_dict=trans_dict_ls[update_i])
            actor_loss += _loss[0] # 因为这个 loss 是负数， 所以给个平方看收敛
            critic_loss += _loss[1]
            # agent.update(trans_dict=trans_dict_0)
            # agent.update(trans_dict=trans_dict_1)
        # break
        writer.add_scalar("return :", episode_return, episode_i) # 
        writer.add_scalar("actor_loss :", actor_loss, episode_i) # 
        writer.add_scalar("critic_loss :", critic_loss, episode_i) # 

def tensorboard_show():
    
    time.sleep(5)

    command = ["tensorboard.exe", "--logdir=./log/ippo"]
    process = subprocess.Popen(args=command)

    print(process.pid)

if __name__ == '__main__':

    bignut = PPO_Clip(state_dim=state_num, hidden_dim=hidden_dim, action_dim=action_num,
                      ac_lr=ac_lr, clip_eps=clip_eps, gae_lambda=gae_lambda,
                      epochs=epochs, gamma=gamma, max_grad_norm=max_grad_norm,
                      entropy_cof=entropy_cof,critic_cof=critic_cof, device=device)
    
    tmp_thread = Thread(target=tensorboard_show, name="data_show_process")
    
    tmp_thread.start()
    

    train_on_policy(env=env, agent=bignut, num_episodes=episode_num)
    

"""
    发现一个问题，动作总是趋向于 [1, 3] 也就是 不变道， 非常奇怪

    有一个地方是, 车的速度只能加速到30，并不是 40 

    暂时找不到 控制速度的位置， 就先这样

    
    参考 IPPO 论文， 给出几个可以改进的地方：
        
        + loss 统一 传播？
        + entropy
        + 优势函数标准化
        + 暂时没有用 价值函数裁剪， 但是可以等价的把 价值函数的 loss 系数减小一点
    
    问题, 在只训练到 2% 的时候 就已经不转向了，收敛到局部最优有点太快了。

    怎么调的慢一点
    
    # 这里的 entropy 确实有效果，训练到中间，看到了有明显的正确决策
    # 接下来, 两个loss 要接着看一下
    # GAE 标准化看一下
    # entropy 也要看一下吧

"""
    