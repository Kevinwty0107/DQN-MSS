# -*- coding: utf-8 -*-

import torch as T
import numpy as np
import torch_geometric as TG
from torch_geometric.data import Data
from copy import deepcopy
from networkx import complement
import networkx as nx

class env():
    """
    Enviroment for MSS
    """
    def __init__(self, train_size,i,episodes):
        """
        train_size: the number of traning graph
        """
        self.name = "stable set"
        self.train_size = train_size
        self.train_pool = None
        self.graph = None
        self._train_pool()
        self.reset(i,episodes)#!
    
    def _train_pool(self):
        """
        Generate traning instances for MSS.
        Graph size ranges from 20.
        ER: edge probability is 0.15
        BA: average degree 4
        """
        self.train_pool = []
        for i in range(self.train_size):
            #N = np.random.choice(range(20, 30), size=1)[0]
            N = 20
            if np.random.rand() > 0.5:
                edge_index = TG.utils.barabasi_albert_graph(num_nodes=int(N), num_edges=4)
                edge_index = edge_index.long()
            else:
                edge_index = TG.utils.erdos_renyi_graph(num_nodes=int(N), edge_prob=0.3)
                edge_index = edge_index.long()
            #N=5
            #edge_index = T.tensor([[0,0,0,1,1,1,2,2,2,3,3,3],[1,2,3,0,2,3,0,1,3,0,1,2]])
            #edge_index = T.tensor([[0,0,1,1,2,2,3,3],[1,3,0,2,1,3,0,2]])
            #edge_index = T.tensor([[0,1,1,2,3,3,4,4],[3,2,4,1,0,4,3,1]])
            graph = Data(num_nodes=N, edge_index=edge_index)
            #graph.edge_w = T.ones((N, 1)) 
            graph.edge_w  = T.ones(edge_index.shape[1],1)
            #print(N)
            graph.mu = T.cat((TG.utils.degree(edge_index[0]).unsqueeze(1),T.ones((N, 2))), dim=1)
            graph.edge_index = graph.edge_index.float()
            self.train_pool.append(graph)
            
        print("Train pool generated! Size={}".format(self.train_size))

    def reset(self,i,episodes):
        """
        Randomly choose a graph instance from train pool
        """
        idx = np.random.choice(range(self.train_size), size=1)[0]
        self.graph = deepcopy(self.train_pool[idx])
        if episodes-i <=20:
            Output_graph_file(i,self.graph.edge_index,self.graph.num_nodes)#!
        self.graph.node_tag = T.zeros((self.graph.num_nodes, 1))
        self.graph.avail = T.ones(self.graph.num_nodes,1)
        self.graph.done = self._done()

        return self.graph

    def step(self, action):
        """
        take action and update
        """
        edge_index = self.graph.edge_index
        edge_index = edge_index.long()
        #print(self.graph.node_tag)
        self.graph.node_tag[action[0]] = 1
        #print(edge_index[0])
        #print(action, edge_index[0]==action[0])
        self.graph.avail[action[0]] = 0
        self.graph.avail[edge_index[1][edge_index[0]==action[0]]]=0     # self.graph.action neighbors = 0
        #print(self.graph.node_tag)
        self.graph.done = self._done()

        reward = 1;

        return reward, self.graph

    def _done(self):
        if sum(self.graph.avail) == 0:
            #return T.tensor([1.0],dtype=T.long)
            return 1.0
        else:
            #return T.tensor([0.0],dtype=T.long)
            return 0.0


class create_graph():#!
    def __init__(self, graph_array):
        self.graph_array = graph_array

    def list_duplicates(self, seq):
        out_list = []
        for i in range(len(seq[0])):
            if [seq[0][i], seq[1][i]] not in out_list and [seq[1][i], seq[0][i]] not in out_list:
                out_list.append((seq[0][i], seq[1][i]))
        return out_list

    def edges(self):
        return(self.list_duplicates(self.graph_array))

    def number_of_edges(self):
        return(int(len(self.graph_array[0])/2))

def Output_graph_file(i,edge_index,num_nodes):#!
    out_edge = edge_index.numpy()
    g = create_graph(out_edge)
    new_g = nx.Graph()
    for (u,v) in g.edges():
        new_g.add_edge(u,v)
    new_g = complement(new_g)
    dimacs_filename = "/Users/wangtaiyi/Documents/Graduate/JHU/2020Spring/Deep Learning in Discrete Optimization/Final_project/graph_testset/mygraph"
    
    with open(dimacs_filename+str(i)+ ".txt", "w") as f:
        # write the header
        f.write("p EDGE {} {}\n".format(num_nodes, new_g.number_of_edges()))
        # now write all edges
        for (u, v) in new_g.edges():
            f.write("e {} {}\n".format(int(u+1), int(v+1)))


    
    

 


