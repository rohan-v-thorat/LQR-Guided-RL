"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from scipy.linalg import expm, sinm, cosm
import math
from operator import matmul
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from variant import *
import tensorflow as tf
import scipy.io as sio
import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import h5py
from scipy.integrate import solve_ivp


class DOS(gym.Env):
    def __init__(self):

        #self.high = np.ones(18)
        #self.low  = np.zeros(18)

        # keep temp = 16 for LQR-Guided RL orelse keep temp = 12
        temp = 16
        self.high = 10*np.ones(temp)
        self.low  = np.zeros(4)
        
        self.action_space = spaces.Box(low=np.array([-1.]), high=np.array([1.]), dtype=np.float32)
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.for_plot=[]
        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action,X_,theta_pre=None):

        nonlinear_para = 1
        m = 1
        k = 100
        c = 0.4

        state = self.state
        state = np.concatenate([[X_],[state]],axis=1)[0]

        input = np.concatenate([[state[0:2]], [action]], axis=1)

        # Dynamic model
        dt = 0.02
        A_c = np.array([[0.,1.],[-k/m,-c/m]])
        B_c = np.array([[0.],[1/m]])

        C_d = np.array([-k/m,-c/m])
        D_d = np.array([1/m])
      
        ground_acc = state[3]
        ground_force = m*np.array([ground_acc])

        def CSSM(t, x, A_c, B_c, action, ground_force, nonlinear_para):
            return matmul(A_c,x) + matmul(B_c,(action - ground_force)) + [0,-nonlinear_para*x[0]**3]

        sol = solve_ivp(CSSM, [0, dt], input[:,:2][0], args=(A_c, B_c, action, ground_force, nonlinear_para),t_eval=[dt])

        x_pre = sol.y
        x_pre = np.transpose(x_pre)[0]

        y_pre = matmul(C_d,np.transpose(input[:,:2]) ) + D_d*action[0] - nonlinear_para*x_pre[0:1]**3

        done = False

        cost = abs(x_pre[0])  + abs(y_pre[0])/100 + abs(action[0])/1000
        return np.transpose(x_pre), cost, done, y_pre

    def reset(self):

        temp = 16 # keep temp = 16 for LQR-Guided RL orelse keep temp = 12
        self.state = self.np_random.uniform(low=-self.high/2, high=self.high/2, size=(temp,))
        self.steps_beyond_done = None
        return np.array(self.state)



