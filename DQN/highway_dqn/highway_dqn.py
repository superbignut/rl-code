from typing import Any
import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import pprint
import numpy as np
# from D_PPO import PPO
from Rainbow_dqn import DQN
import torch 
from tqdm import tqdm
import collections
import random

np.set_printoptions(linewidth=1000,suppress=True)


env = gym.make('highway-v0',render_mode='human')
env.configure({
    "lanes_count": 4,
    "vehicles_count": 50,
    "vehicles_density": 0.7,
    "simulation_frequency": 20,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "normalize_reward": True,
    "collision_reward": -1, # 超参
    "lane_change_reward": -0.05, # 变道惩罚
    "reward_speed_range": [20, 30],  # 实际车速范围
    "initial_lane_id": None,
    "action": {
        "type": "DiscreteMetaAction"
    },
    "observation": {
        "type": "Kinematics",
        'lanes_count': 4,
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
        "order": "sorted" # "order"，"shuffled"
    }
})

env.reset(seed=0)

class ReplayBuffer():
    def __init__(self,capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity) # 先进先出 
    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done
    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    
episode_num = 40000
batch_num = 1 # buffer，不使用batch
epslon = 0.1 # 改为0.1
epslon_des = 0.999 # epslon衰减
hidden_dim = 256 
gamma = 0.98
buffer_batch_size =  256 # 对应修改
buffer_minimal_size = 512 # 超参
buffer_size = 10000 # 超参
optim_lr = 1e-4 # 超参
target_update = 10 # 超参

# reward_list = []
if torch.cuda.is_available():
    print("Backend is cuda.")
else:
    print("Backend is cpu.")

device = torch.device('cuda')

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n

model_name = 'DQN'
agent = DQN(state_dim=state_dim, hidden_dim=hidden_dim,action_dim=action_dim,device=device,epslon=epslon,\
            gamma=gamma,lr=optim_lr,target_update=target_update,epslon_des=epslon_des)

replay_buffer = ReplayBuffer(buffer_size)

for index in tqdm(range(episode_num//batch_num)):
    # batch_reward = 0
    # with open('./save/data.txt', 'a') as f:
    for _ in range(batch_num): 
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.take_action(state=state)
            # np.savetxt(f, state, fmt='%.3f')
            # f.write('\n')
            state_next, reward, done, trunc, _  = env.step(action)
            print(state)
            if trunc == True: 
                done = True
            replay_buffer.add(state=state,action=action,reward=reward,next_state=state_next,done=done) 
            state = state_next
            # batch_reward += reward
            # print(reward)
        # 这里可以改成 一局game 更新一次 每一帧更新一次，有点太频繁了
        if replay_buffer.size() >= buffer_minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(buffer_batch_size)
            sarsd_dict = {'state':b_s,
                            'action':b_a,
                            'reward':b_r,
                            'state_next':b_ns,
                            'done':b_d
                            }
            agent.update(sarsd_dict)
    # reward_list.append(batch_reward)
    if index % 5000 == 0:
        addr = './save/dqn' + str(index) + '_para.pth'
        torch.save(agent.qnet.state_dict(), addr)
env.close()

