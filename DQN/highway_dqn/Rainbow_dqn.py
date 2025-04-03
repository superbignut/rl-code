import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
###########################################
'''
attentiond的参数直接在这里修改
'''
buffer_batch_size = 256 # 对应修改
attn_dim_n = 12 # 对应修改
attn_dim_in = 7 # 对应修改
attn_dim_k = 32 # 超参
attn_dim_v = 32 # 超参
attn_dim_head = 2 # 超参
mask_cons = -10000.0 # 超参
###########################################

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, batch_size = buffer_batch_size, \
                 dim_n = attn_dim_n, dim_in = attn_dim_in, dim_k = attn_dim_k, dim_v = attn_dim_v, num_heads=attn_dim_head) -> None:
        super().__init__()
        # 定义线性变换矩阵
        self.dim_in = dim_in
        self.batch_size = batch_size
        self.nh = num_heads
        self.nk = attn_dim_k
        self.nv = attn_dim_v


        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) 
        self.fc_mid = torch.nn.Linear(hidden_dim, hidden_dim) # 增加 attn_fc 去掉mic_fc
        self.fc_2 = torch.nn.Linear(hidden_dim, hidden_dim) # 增加 attn_fc 去掉mic_fc
        self.fc_a = torch.nn.Linear(hidden_dim, action_dim) # FC_A
        self.fc_v = torch.nn.Linear(hidden_dim, 1)  # FC_v


    def forward(self, x):
        # print(x)
        batch, n, dim_in = x.shape
        x = x.view(batch, -1)

        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc_mid(x)) 
        X = F.relu(self.fc_2(x))
        A = self.fc_a(x)
        V = self.fc_v(x) # 这里是将V加到了A的每一列上
        Q = V + A - A.mean(dim=1).view(-1,1) # dueling dqn
        return Q
    
class DQN():
    def __init__(self, state_dim, hidden_dim, action_dim, device, epslon, gamma, lr, target_update, epslon_des) -> None:
        self.qnet = Qnet(state_dim=state_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device=device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),lr=lr)

        self.target_qnet = Qnet(state_dim=state_dim,hidden_dim=hidden_dim,action_dim=action_dim).to(device=device)
        
        self.action_dim = action_dim
        self.device = device
        self.epslon = epslon
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update  # 目标网络更新频率
        self.epslon_des = epslon_des
        self.count = 0  # 计数器,记录更新次数
        self.loss_list = []
    def take_action(self, state, wich='normal'):
        state = np.array([state]) # 包一层
        if wich == 'max':
            state = torch.tensor(state).to(self.device)
            action = self.qnet(state).argmax().item() # 返回最大的q值的下标
            return action
        
        if np.random.random() < self.epslon:
            action = np.random.randint(self.action_dim)
        else:
            if np.random.random() < 0.001: # 增大一层探索的机会，并且不参与衰减
                action = np.random.randint(self.action_dim)
                return action
            state = torch.tensor(state).to(self.device)
            action = self.qnet(state).argmax().item() 
        return action

    

    def update(self, sarsd_dict):
        state = np.array(sarsd_dict['state'])
        state = torch.tensor(state).to(self.device) 
        reward = torch.tensor(sarsd_dict['reward'],dtype=torch.float).view(-1,1).to(self.device)
        action = torch.tensor(sarsd_dict['action']).view(-1,1).to(self.device)
        dones = torch.tensor(sarsd_dict['done'],dtype=torch.float).view(-1, 1).to(self.device)
        state_next = np.array(sarsd_dict['state_next'])
        state_next = torch.tensor(state_next).to(self.device)

        q_values = self.qnet(state).gather(1, action)
        max_action = self.qnet(state_next).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_qnet(state_next).gather(1, max_action) # Double DQN

        q_targets = reward + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets.detach())) # 增加detach()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        # self.loss_list.append(dqn_loss.cpu().detach().item())
        if self.count % self.target_update == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())  # 更新目标网络 
            # 目标网络更新 epslon衰减
            self.epslon = self.epslon * self.epslon_des
            print('\n',self.count) # 每次更新目标网络输出
        self.count += 1
        
        
        