from icecream import ic
import tensorflow as tf
import numpy as np
import time
from .squash_bijector import SquashBijector
from .utils import evaluate_training_rollouts
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
sys.path.append("..")
from robustness_eval import training_evaluation
from scipy.io import savemat
from scipy.io import loadmat
from pool.pool import Pool
import logger
from variant import *
import scipy.io as sio
from sklearn import preprocessing
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
SCALE_DIAG_MIN_MAX = (-10, 1)
SCALE_lambda_MIN_MAX = (0, 1)
SCALE_alpha_MIN_MAX = (0, 10)
SCALE_beta_MIN_MAX = (0, 1)

class LAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,


                 variant,

                 action_prior = 'uniform'
                 ):


        ###############################  Model parameters  ####################################
        # self.memory_capacity = variant['memory_capacity']

        self.batch_size = variant['batch_size']
        gamma = variant['gamma']

        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        
        # self.pointer = 0
        self.sess = tf.Session()
        self._action_prior = action_prior
        s_dim = s_dim * (variant['history_horizon']+1)
        self.a_dim, self.s_dim, = a_dim, s_dim
        self.history_horizon = variant['history_horizon']
        self.working_memory = deque(maxlen=variant['history_horizon']+1)
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy =  -self.a_dim   #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy
        self.finite_horizon = variant['finite_horizon']
        self.soft_predict_horizon = variant['soft_predict_horizon']
        with tf.variable_scope('Actor'):
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
            self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')
            self.R_N_ = tf.placeholder(tf.float32, [None, 1], 'r_N_')
            self.V = tf.placeholder(tf.float32, [None, 1], 'v')
            self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
            self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
            self.LR_lag = tf.placeholder(tf.float32, None, 'LR_lag')
            self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
            self.LR_L = tf.placeholder(tf.float32, None, 'LR_L')
            # self.labda = tf.placeholder(tf.float32, None, 'Lambda')
            labda = variant['labda']
            alpha = variant['alpha']
            alpha3 = variant['alpha3']
            beta = variant['beta']
            log_beta = tf.get_variable('beta', None, tf.float32, initializer=tf.log(beta))
            log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda))
            log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature
            # The update is in log space
            self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
            #self.alpha = tf.clip_by_value(tf.exp(log_alpha), *SCALE_alpha_MIN_MAX)
            self.alpha = tf.exp(log_alpha)
            #self.beta = tf.exp(log_beta)
            self.beta = tf.clip_by_value(tf.exp(log_beta), *SCALE_beta_MIN_MAX)

            self.a, self.deterministic_a, self.a_dist = self._build_a(self.S)  # 这个网络用于及时更新参数
            s_a, d_a, a_dis = self._build_a(self.S_,reuse=True)

            self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络


            self.use_lyapunov = variant['use_lyapunov']
            self.adaptive_alpha = variant['adaptive_alpha']
            self.constraint = variant['constraint']

            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/actor')
            l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/Lyapunov')

            ###############################  Model Learning Setting  ####################################
            ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))
            target_update = [ema.apply(a_params),  ema.apply(l_params)]  # soft update operation

            # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
            lya_a_, _, lya_a_dist_ = self._build_a(self.S_, reuse=True)

            self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

            # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度

            l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            # lyapunov constraint
            # energy decreasing
            self.l_derta = tf.reduce_mean(self.l_ - self.l + (alpha3) * self.R)

            labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
            self.l_action = tf.reduce_mean(tf.norm(d_a-self.deterministic_a))
            alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
            beta_loss = -tf.reduce_mean(log_beta * tf.stop_gradient(self.l_action-0.1))
            self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)
            self.lambda_train = tf.train.AdamOptimizer(self.LR_lag).minimize(labda_loss, var_list=log_labda)
            self.beta_train = tf.train.AdamOptimizer(0.01).minimize(beta_loss, var_list=log_beta)

            if self._action_prior == 'normal':
                policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(self.a_dim),
                    scale_diag=tf.ones(self.a_dim))
                policy_prior_log_probs = policy_prior.log_prob(self.a)
            elif self._action_prior == 'uniform':
                policy_prior_log_probs = 0.0

            if self.use_lyapunov is True:
                # The l_derta, the smaller the better
                a_loss = self.labda * self.l_derta + self.alpha * tf.reduce_mean(log_pis) - policy_prior_log_probs + beta*self.l_action #+ self.R
            else:
                a_loss = a_preloss

            self.a_loss = a_loss
            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)

            next_log_pis = a_dist_.log_prob(a_)
            with tf.control_dependencies(target_update):  # soft replacement happened at here
                if self.approx_value:
                    if self.finite_horizon:
                        if self.soft_predict_horizon:
                            l_target = self.R - self.R_N_ + tf.stop_gradient(l_)
                        else:
                            l_target = self.V
                    else:
                        l_target = self.R + gamma * (1-self.terminal)*tf.stop_gradient(l_)  # Lyapunov critic - self.alpha * next_log_pis
                else:
                    l_target = self.R

                self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
                self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.l_error, var_list=l_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.diagnotics = [self.labda, self.alpha, self.l_error, tf.reduce_mean(-self.log_pis), self.a_loss]

            if self.use_lyapunov is True:
                self.opt = [self.ltrain, self.lambda_train]
            self.opt.append(self.atrain)
            if self.adaptive_alpha is True:
                self.opt.append(self.alpha_train)
            if self.constraint is True:
                self.opt.append(self.beta_train)

    def choose_action(self, s, evaluation = False):
        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        if evaluation is True:
            try:
                return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
            except ValueError:
                return
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_C, LR_L,LR_lag, batch):

        bs = batch['s']  # state
        ba = batch['a']  # action

        br = batch['r']  # reward

        bterminal = batch['terminal']
        bs_ = batch['s_']  # next state
        feed_dict = {self.a_input: ba,  self.S: bs, self.S_: bs_, self.R: br, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L, self.LR_lag:LR_lag}
        if self.finite_horizon:
            bv = batch['value']
            b_r_ = batch['r_N_']
            feed_dict.update({self.V:bv, self.R_N_:b_r_})

        self.sess.run(self.opt, feed_dict)
        labda, alpha, l_error, entropy, a_loss = self.sess.run(self.diagnotics, feed_dict)

        return labda, alpha, l_error, entropy, a_loss

    def store_transition(self, s, a,d, r, l_r, terminal, s_):
        transition = np.hstack((s, a, d, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, name='actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]
            squash_bijector = (SquashBijector())
            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)  
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)  
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.leaky_relu, name='l3', trainable=trainable) 

            
            mu = tf.layers.dense(net_2, self.a_dim, activation= None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_2, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)


            bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
            raw_action = bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            ## Construct the distribution
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=mu,
                    scale_diag=sigma),
            ))
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector)

            clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution


    def evaluate_value(self, s, a):

        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        return self.sess.run(self.l, {self.S: s[np.newaxis, :], self.a_input: a[np.newaxis, :]})[0]

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.leaky_relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)  
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.leaky_relu, name='l3', trainable=trainable)  
            return tf.expand_dims(tf.reduce_sum(tf.square(net_2), axis=1),axis=1)  # Q(s,a)


    def save_result(self, path):

        save_path = self.saver.save(self.sess, path + "/policy/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/')
        if model_file is None:
            success_load = False
            print("Load failed, model file:", model_file)
            print("#########################################################")
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        print("Load successful, model file:", model_file)
        print("#########################################################")
        return success_load

def LQR_policy(x):
    K = np.array([106.6,21.5]) #0.2760,9.6233 #1.3749,0.5833
    control_force = sum(-K*x)
    return control_force            

def train(variant):
    store_test_reward = []

    Min_cost=1000000


    print("Data Got!")
    ##################Normalizing the data#####################
    scaler = preprocessing.MinMaxScaler()

    print("Data Normalized!")


    env_name = variant['env_name']
    env = get_env_from_name(env_name)
    controller = variant['controller']
    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']


    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    batch_size = policy_params['batch_size']

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic

    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]


    a_dim = env.action_space.shape[0] # a_dim = 4
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    policy = LAC(a_dim,s_dim, policy_params)


    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value  # dim.value is used in TensorFlow 1.x
        total_parameters += variable_parameters

    print("Total trainable parameters:", total_parameters)
    ##### Note: the parameter are getting affected inspite of removing from the computational graph


    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1, # disturbance dimensions
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        'history_horizon': policy_params['history_horizon'],
        'finite_horizon':policy_params['finite_horizon']
    }
    if 'value_horizon' in policy_params.keys():
        pool_params.update({'value_horizon': policy_params['value_horizon']})
    else:
        pool_params['value_horizon'] = None
    pool = Pool(pool_params)
    # For analyse
    Render = env_params['eval_render']

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('tau', policy_params['tau'])

    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    logger.logkv('target_entropy', policy.target_entropy)

    stop_code = 'NO'
    for i in range(max_episodes):
        
        if stop_code == 'YES':
            break
        
        current_path = {'rewards': [],
                        'a_loss': [],

                        'alpha': [],

                        'lambda': [],
                        'lyapunov_error': [],
                        'entropy': [],

                        }

        if global_step > max_global_steps:
            break

        s = env.reset() 
       
        traj = sio.loadmat('data/gacc_train.mat')
        traj = traj['T']
        start_point = 0
        if 'LQR_Guided_RL' in controller:
            s = np.array([0.,traj[start_point,0],0.,LQR_policy(np.array([0,0])),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        elif 'RL' in controller:
            s = np.array([0.,traj[start_point,0],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        env.state = s
        env.state = s

        ic(i)#print(i)
        # print(i)

        max_ep_steps = 1001 #3961 #1001
        for j in range(start_point+1,start_point+1+max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s,False)

            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            action_temp = action.copy()
            

            if j==1:
                X_ = np.array([0.,0.])

            if 'LQR_Guided_RL' in controller:
                action_base = 0.5*LQR_policy(X_)
            elif 'RL' in controller:
                action_base = 0*LQR_policy(X_)
            action += action_base

            # Run in simulator
            X_, r, done, y_pre = env.step(action,X_) # 'X_'size 8 

            if 'LQR_Guided_RL' in controller:
                s_ = np.concatenate([y_pre,[traj[j,0]],action_temp,[LQR_policy(X_)],s[:12]], axis=0)
            elif 'RL' in controller:
                s_ = np.concatenate([y_pre,[traj[j,0]],a,s[:9]], axis=0)  

            env.state = s_

            if training_started:
                global_step += 1

            if j == max_ep_steps - 1+start_point:
                done = True

            terminal = 1. if done else 0.
            pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_)
            
            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0: # steps_per_cycle = 1, min_memory_size = 1000 
                training_started = True
                
                for _ in range(train_per_cycle):  # train_per_cycle = 1
                    batch = pool.sample(batch_size) # avoid taking samples from extreme past
                    labda, alpha, l_loss, entropy, a_loss = policy.learn(lr_a_now, lr_c_now, lr_l_now, lr_a_now, batch)

            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:
                print(global_step)
                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    eval_diagnotic = training_evaluation(variant, env, policy, LQR_policy)
                    [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                    training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                     for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))
                    print("Current lowest cost:", Min_cost)
                    store_test_reward.append(eval_diagnotic['test_return'])
                    # also test the policy at time t = 0
                    # break
                logger.dumpkvs()
                
                if eval_diagnotic['test_return']  <= Min_cost:
                    Min_cost = eval_diagnotic['test_return'] 
                    print("New lowest cost:", Min_cost)
                    if 'LQR_Guided_RL' in controller:
                        policy.save_result(log_path)

            s = s_


            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)
                
                # frac = 1/(1+j)
                if 1*global_step < max_global_steps:  # 20
                    frac = 1.0 - (1*global_step - 1.0) / max_global_steps
                    lr_a_now = lr_a * frac  # learning rate for actor
                    lr_c_now = lr_c * frac  # learning rate for critic
                    lr_l_now = lr_l * frac  # learning rate for critic

                break
                    
    print('Running time: ', time.time() - t1)
    return

def eval(variant):
    env_name = variant['env_name']

    Min_cost=1000000
    #with h5py.File('submit_version_data_8_4_2_8.h5', 'r') as hdf:
        #data = hdf.get('Data')
        #data = np.array(data)

    print("Data Got!")
    ##################Normalizing the data#####################
    scaler = preprocessing.MinMaxScaler()
    #scaler.fit(data)

    print("Data Normalized!")

    # env = get_env_from_name(env_name)
    env = get_env_from_name('Duffing_oscillator_sim')
    controller = variant['controller']
    env_params = variant['env_params']
    max_ep_steps = env_params['max_ep_steps']
    max_ep_steps = 1001 #3961 #1001
    policy_params = variant['alg_params']
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = LAC(a_dim, s_dim, policy_params)
    print(s_dim)
    log_path = variant['log_path'] + '/eval/' + str(0)
    logger.configure(dir=log_path, format_strs=['csv'])
    print(variant['log_path'])
    
    policy.restore(variant['log_path'] + '/' + str(0)+'/policy')
    
    # Training setting
    t1 = time.time()
    PLOT_action = []
    PLOT_ground_theta = []
    PLOT_x = []
    PLOT_ground_x = []
    PLOT_lqr_force = []
    PLOT_rl_force = []

    mst=[]
    agent_traj=[]
    ground_traj=[]

    SAVESTEPS=[]
    SAVETRAJS=[]
    FLIGHTNUM=1
    for i in range(1):
        cost = 0
        traj_num = 0

        data = sio.loadmat('data/gacc_test.mat')
        data = data['T']
        # print(data.keys())
        traj = np.concatenate([data[:,:]], axis=1)

        start_point = 0
        s = np.array([0.,traj[start_point,0],0.,LQR_policy(np.array([0,0])),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        env.state = s

        for j in range(start_point+1,start_point+1+max_ep_steps):

            if agent_traj == []:
                agent_traj = s[0:2]
            else:
                agent_traj = np.vstack((agent_traj, s[0:2]))

            if j==1:
                X_ = np.array([0.,0.])
            
            a = policy.choose_action(s,True)   # Takes deterministic action value

            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            action_temp = action.copy()

            action_base = 0.5*LQR_policy(X_)
            action += action_base           

            X_, r, done, y_pre = env.step(action,X_)

            s_ = np.concatenate([y_pre,[traj[j,0]],action_temp,[LQR_policy(X_)],s[:12]], axis=0)
            
            if j == start_point +1:
                print(r)
            env.state = s_

            if PLOT_action == []:
                PLOT_action = action
            else:
                PLOT_action = np.vstack((PLOT_action,action))

            if PLOT_lqr_force == []:
                PLOT_lqr_force = [action_base]
            else:
                PLOT_lqr_force = np.vstack((PLOT_lqr_force,[action_base]))

            if PLOT_rl_force == []:
                PLOT_rl_force = action_temp
            else:
                PLOT_rl_force = np.vstack((PLOT_rl_force,action_temp))

            # Plot tracking

            if PLOT_x == []:
                PLOT_x = np.hstack((X_[0:2],y_pre[0]))
            else:
                PLOT_xx = np.hstack((X_[0:2],y_pre[0]))
                PLOT_x = np.vstack((PLOT_x, PLOT_xx))

            logger.logkv('rewards', r)
            logger.logkv('timestep', j)
            logger.dumpkvs()

            cost = cost + r

            if j == max_ep_steps - 1 + start_point:
                done = True

            s = s_

            if done:

                break
    x = np.linspace(0,j-1,j )
    # print(j)
    print(cost)

    disp = {'displacement':PLOT_x[:,0]}
    savemat("data/disp_LQR_Guided_RL.mat",disp )

    vel = {'velocity':PLOT_x[:,1]}
    savemat("data/vel_LQR_Guided_RL.mat",vel )

    acc = {'acceleration':PLOT_x[:,2]}
    savemat("data/acc_LQR_Guided_RL.mat",acc )

    force = {'force':PLOT_action}
    savemat("data/force_LQR_Guided_RL.mat",force )

    disp2 = loadmat('data/disp_LQR.mat')['displacement']
    vel2 = loadmat('data/vel_LQR.mat')['velocity']
    acc2 = loadmat('data/acc_LQR.mat')['acceleration']
    force2 = loadmat('data/force_LQR.mat')['force']

    plt.figure(1)
    plt.subplot(3, 1, 1) 
    plt.plot(x, force2, color='C1', label='LQR')
    plt.plot(x, PLOT_action[:], color='C0', label='LQR-Guided RL',linestyle='--') 
    plt.legend(loc='best') 
    plt.ylabel('control force')

    plt.subplot(3, 1, 2)
    plt.plot(x, PLOT_lqr_force[:], color='C0', label='LQR component')
    plt.legend(loc='best') 
    plt.ylabel('control force')    

    plt.subplot(3, 1, 3)   
    plt.plot(x, PLOT_rl_force[:], color='C0', label='RL component')
    plt.legend(loc='best') 
    plt.ylabel('control force')    
    plt.savefig('plots/force.png', dpi=300)

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(x, disp2[0,:], color='C1', label='LQR')
    plt.plot(x, PLOT_x[:, 0], color='C0', label='LQR-Guided RL',linestyle='--') 
    plt.legend(loc='best') 
    plt.ylabel('displacement')

    plt.subplot(3,1,2)
    plt.plot(x, vel2[0,:], color='C1', label='LQR')
    plt.plot(x, PLOT_x[:, 1], color='C0', label='LQR-Guided RL',linestyle='--')
    plt.legend(loc='best')
    plt.ylabel('velocity')

    plt.subplot(3, 1, 3)
    plt.plot(x, acc2[0,:], color='C1', label='LQR')
    plt.plot(x, PLOT_x[:, 2], color='C0', label='LQR-Guided RL',linestyle='--')  
    plt.legend(loc='best') 
    plt.ylabel('acceleration')

    plt.savefig('plots/response.png', dpi=300)
    
