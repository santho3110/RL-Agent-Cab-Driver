#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import routines
import math
import random
import numpy as np
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

class CabDriver():
    
    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = list(product(range(m), range(t), range(d)))
        self.state_init = self.state_space[np.random.choice(len(self.state_space))]
        self.hours_left = 30*24
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # (current-location, time, day)
        state_encod = np.zeros(m+t+d, int)
        state_encod[[state[0], state[1]+m, state[2]+m+t]] = 1
        return state_encod
    
    
    def state_batch_encode_archII(self, state_batch):
        """convert the state batch into a vector so that it can be fed to the NN."""
        # (batch, (current-location, time, day))
        return np.append(np.append(np.eye(m)[state_batch[:, 0]],                          np.eye(t)[state_batch[:, 1]], axis=1),                          np.eye(d)[state_batch[:, 2]], axis=1)
    

    # Use this function if you are using architecture-1 
    def state_encod_arch1(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        # (current-location, time, day) , (pick, drop)
        state_encod = np.zeros(m+t+d+m+m, int)
        state_encod[[state[0], state[1]+m, state[2]+m+t, action[0]+m+t+d, action[1]+m+t+d+m]] = 1
        return state_encod


    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        if requests >15:
            requests =15
            
        possible_actions_idx = [0]+sorted(random.sample(range(1, (m-1)*m +1), requests))
        actions = [self.action_space[i] for i in possible_actions_idx]

        return possible_actions_idx, actions   
    

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if sum(action)==0:
            return -C
        t = Time_matrix[action[0]][action[1]][state[1]][state[2]]
        reward = R*t - C*t
        if state[0]!=action[0]:
            tp = Time_matrix[state[0]][action[0]][state[1]][state[2]]
            reward-=tp
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        # trip_time[1].append(self.hours_left) Track Trip Time 
        if sum(action)==0:
            time = 1
            updated_state = state[0]
        else:
            time = Time_matrix[action[0]][action[1]][state[1]][state[2]]
            updated_state = action[1]
        updated_time = int(state[1]+time)
        updated_day = state[2]
        if updated_time>=t:
            updated_time = updated_time-t
            updated_day = state[2]+1
        if updated_day>=d:
            updated_day = 0
        self.hours_left-=time
        terminal = self.hours_left<=0
        return (updated_state, updated_time, updated_day), terminal
    

    def reset(self):
        return self.action_space, self.state_space, self.state_init

