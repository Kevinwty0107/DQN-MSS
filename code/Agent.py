# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:54:06 2020

"""

from Q import Q_Fun
from ReplayBuffer import replayBuffer

import torch
import torch.nn as nn
from torch_scatter import scatter_max
import numpy as np

class Agent(nn.Module):
    def __init__(self, 
                 epsilon=1.0, 
                 eps_decay = 1E-4,
                 gamma=1,
                 batch_size=20, 
                 lr=0.01,
                 lr_gamma=0.95,
                 in_dim=3, 
                 hid_dim=64, 
                 T=5,
                 mem_size=100000, 
                 test=False,
                 replace_target = 100):
        super(Agent, self).__init__()        
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.T = T
        self.mem_size = mem_size
        self.test = test
        
        self.pre_Q = Q_Fun(in_dim, hid_dim, T, lr, lr_gamma)
        self.target_Q = Q_Fun(in_dim, hid_dim, T, lr, lr_gamma)
        self.memory = replayBuffer(mem_size)

        self.learn_step_cntr = 0
        self.replace_target = replace_target

        self.loss_list = []
        
    def choose_action(self, graph):
        graph = graph.to(self.pre_Q.device)
        Q_value = self.pre_Q.forward(graph)
        # make sure select new nodes
        if np.random.rand() < self.epsilon and not self.test:
            self.epsilon = max(0.05, self.epsilon - self.eps_decay)
            while True:
                action = np.random.choice(graph.num_nodes, size=1)
                if graph.avail[int(action)] == 1:
                    break
        else:
            _, q_action = torch.sort(Q_value, descending=True)
            for action in q_action:
                if graph.avail[action] == 1:
                    break

        return [int(action.item())]
    
    def remember(self, *args):
        self.memory.store(*args)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.learn_step_cntr += 1
        #print('learn')
        if self.learn_step_cntr % self.replace_target == 1:
            self.target_Q.load_state_dict(self.pre_Q.state_dict())
 
        graphs_pre, graphs_later, actions, rewards, num_nodes = self.memory.sample(self.batch_size)
        #print('graphs pre',graphs_pre.num_nodes)
        graphs_pre   = graphs_pre.to(self.pre_Q.device)
        graphs_later = graphs_later.to(self.pre_Q.device)
        actions      = actions.to(self.pre_Q.device)
        #print(actions)
        rewards      = rewards.to(self.pre_Q.device)

        self.pre_Q.optimizer.zero_grad()
        self.target_Q.optimizer.zero_grad()
        
        finished = 1.0-graphs_later.done;
        y_target = rewards + self.gamma * finished * self._max_Q(graphs_later)
        #y_target = rewards + self.gamma * self._max_Q(graphs_later)
        y_pred   = self.pre_Q(graphs_pre)[self._idx(actions, num_nodes)]
        loss     = torch.mean(torch.pow(y_target-y_pred, 2))
        
        self.loss_list = []
        self.loss_list.append(loss.item())
        
        loss.backward()
        
        self.pre_Q.optimizer.step()
        self.target_Q.optimizer.step()



    def _max_Q(self, graphs):
        batch = graphs.batch
        #Q_value = self.target_Q(grahs)
        Q_value = self.target_Q.forward(graphs)
        return scatter_max(Q_value, batch)[0].detach()

    def _idx(self, actions, num_nodes):
        """
        adjust actions to batch
        """
        for i in range(len(actions)-1):
            actions[i+1] += int(sum(num_nodes[:i+1]))
        
        return actions

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def loss_record(self):
        return self.loss_list