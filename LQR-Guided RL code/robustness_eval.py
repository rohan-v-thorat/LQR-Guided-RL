import tensorflow as tf
import os
from variant import *

import numpy as np
import time
import logger
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
import h5py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def training_evaluation(variant, env, policy, LQR_policy):
    # fig = plt.figure()

    ###################Normalizing the data#####################
    scaler = preprocessing.MinMaxScaler()
    #scaler.fit(data)


    env_name = variant['env_name']
    controller = variant['controller']
    env_params = variant['env_params']

    max_ep_steps = env_params['max_ep_steps']


    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    # For analyse
    Render = env_params['eval_render']

    # Training setting

    total_cost = []
    death_rates = []

    episode_length = []

    die_count = 0
    seed_average_cost = []

    print("Test on trajectory:", 1500)
    for i in range(variant['store_last_n_paths']):

        cost = 0
        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        traj = sio.loadmat('data/gacc_train.mat')
        traj = traj['T']
        start_point = 0

        if 'LQR_Guided_RL' in controller:
            s = np.array([0.,traj[start_point,0],0.,LQR_policy(np.array([0,0])),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        elif 'RL' in controller:
            s = np.array([0.,traj[start_point,0],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        env.state = s

        max_ep_steps = 1001 #3961 #1001
        for j in range(start_point+1,max_ep_steps+start_point+1):

            if Render:
                env.render()
            a = 1*policy.choose_action(s, True)
            if j==1:
                X_ = np.array([0.,0.])

            action= a_lowerbound/1 + (a + 1.) * (a_upperbound/1 - a_lowerbound/1) / 2
            action_temp = action.copy()
            if 'LQR_Guided_RL' in controller:
                action_base = 0.5*LQR_policy(X_)
            elif 'RL' in controller:
                action_base = 0*LQR_policy(X_)
            action += action_base

            # if j <= start_point+1+3:
            #     print(action)
            X_, r, done, y_pre= env.step(action,X_)
            # print(distance)
            if 'LQR_Guided_RL' in controller:
                s_ = np.concatenate([y_pre,[traj[j,0]],action_temp,[LQR_policy(X_)],s[:12]], axis=0)
            elif 'RL' in controller:
                s_ = np.concatenate([y_pre,[traj[j,0]],a,s[:9]], axis=0)            

            if j == start_point +1:
                print(r)
            env.state = s_

            cost += r

            if j == max_ep_steps+start_point - 1:
                done = True
            s = s_

            if done:
                seed_average_cost.append(cost)
                episode_length.append(j-start_point)
                if j < max_ep_steps-1:
                    die_count += 1
                break
    death_rates.append(die_count/(i+1)*100)
    total_cost.append(np.mean(seed_average_cost))
    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)
    death_rate = np.mean(death_rates)
    death_rate_std = np.std(death_rates, axis=0)
    average_length = np.average(episode_length)
    diagnostic = {'test_return': total_cost_mean,
                  'test_average_length': average_length}
    # print('cost:',cost)
    # print('total_cost:',total_cost)
    print('total_cost_mean:',total_cost_mean)
    # print(cost)
    return diagnostic

