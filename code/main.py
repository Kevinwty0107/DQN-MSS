# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:51:15 2020

"""
from Env import env
from Agent import Agent
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import networkx as nx
from networkx import complement
from Env import create_graph
import collections

def Greedy(edge_index):
    """Manually Greedy for MSS"""
    degree = collections.defaultdict(list)
    node_candidate = dict()
    greedy_mss = []
    for i in range(edge_index.shape[1]):
        u,v = int(edge_index[0,i]), int(edge_index[1,i])
        degree[u].append(v)
        node_candidate[u] = 1
    sorted_d = sorted(degree.items(),key=lambda x: len(x[1]))
    for node,s in sorted_d:
        if node_candidate[node] == 1:
            greedy_mss.append(node)
            for neighbor in s:
                node_candidate[neighbor] = 0 # change to not available
    return greedy_mss

def MSS(edge_index):
    """Networkx Greedy for MSS"""
    out_edge = edge_index.numpy()
    g = create_graph(out_edge)
    new_g = nx.Graph()
    for (u,v) in g.edges():
        new_g.add_edge(u,v)
    new_g = complement(new_g)
    SS_list = list(nx.find_cliques(new_g))
    MSS_list = max(SS_list,key=len)
    return MSS_list


if __name__ == "__main__":
    num_episodes = 20000 #! Number could be changed
    env = env(train_size=40,i=1,episodes=num_episodes) #!
    n_step = 4
    agent = Agent()
    scores = []
    loss_list = []
    """
    Please note that after adding this Greedy Algorithm, your running time will be much longer!!! 
    """
    Greedy_filename = "greedy.txt"
    file_greedy = open(Greedy_filename,"w")
    score_filename = "score.txt"#!
    file_score = open(score_filename,"w")#!
    for i in tqdm(range(num_episodes)):
        score = 0
        graph = env.reset(i,num_episodes)#!
        file_greedy.write("Episode: {:<4}, size_greedy: {:<4}, node_num:{:<4}, stable_set:{}\n".
        format(i, len(Greedy(graph.edge_index)),graph.num_nodes, Greedy(graph.edge_index)))
        state_steps = [graph.node_tag]
        
        reward_steps = []
        action_steps = []
        steps_cntr = 0

        while not graph.done:
            graph_pre = deepcopy(graph)
            action = agent.choose_action(graph)
            #_, _, _, reward, new_state, done = env.step(action)
            reward, graph = env.step(action)
            graph_later = deepcopy(graph)
            #state_steps.append(new_state)
            state_steps.append(graph.node_tag)
            reward_steps.append(reward)
            action_steps.append(action)
            steps_cntr += 1
            
            if steps_cntr > n_step+1:
                #agent.remember(num_nodes,mu,edge_index,edge_w,state_steps[-(n_step+1)],action_steps[-n_step],[sum(reward_steps[-n_step:])],state_steps[-1],done)
                agent.remember(graph_pre, action, reward, graph_later)
                agent.learn()
                loss_list.extend(agent.loss_record())
            state = graph.node_tag
            
        score = len(set([a[0] for a in action_steps]))
        scores.append(score)
  
        file_score.write("Episode: {:<4}, score: {:<4}, node_num:{:<4}, stable_set:{}\n".
                format(i, score, graph.num_nodes, set([a[0] for a in action_steps]))) #!
        """
        you can print your result here
        """
        #print(set([a[0] for a in action_steps]))
        #print("Episode: {:<4}, score: {:<4}, node_num:{:<4}".
        #        format(i, score, graph.num_nodes))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.log(loss_list))
    ax.set_xlabel('Times of learning')
    ax.set_ylabel('Log Loss')
    ax.set_title('Loss Curve')
    plt.show()
    file_greedy.close()
    file_score.close()#!
