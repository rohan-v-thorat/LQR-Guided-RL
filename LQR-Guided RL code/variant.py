
import datetime
import numpy as np
SEED = None

VARIANT = {

    'env_name': 'Duffing_oscillator_sim',
    #training prams
    'algorithm_name': 'LAC',
    'additional_description': '-rebutal-1',
    'controller': 'LQR_Guided_RL', #'LQR_Guided_RL', 'RL'
    
    # 'train': True,
    'train': False,
    #
    'num_of_trials': 1, #10,   # number of random seeds
    # 'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'store_last_n_paths': 1,  # number of trajectories for evaluation during training
    'start_of_trial': 0,

    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 2000,
}
if VARIANT['algorithm_name'] == 'RARL':
    ITA = 0
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

ENV_PARAMS = {
    'Duffing_oscillator_sim': {
        'max_ep_steps': 1000,
        'max_global_steps': int(1e7),
        'max_episodes':  int(1e2),
        'disturbance dim': 8,
        'eval_render': False, },
}
ALG_PARAMS = {
    'LAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000*1e0,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.*1e0,
        'alpha3': 0.995,
        #'alpha3':1.,
        'beta': 0.,
        'tau': 5e-3,
        'lr_a': 1e-4*1e0,
        'lr_c': 3e-4*1e0,
        'lr_l': 3e-4*1e0,
        'gamma':0.999,
        #'gamma': 0.995,
        #'gamma': 0.75,
        #'gamma':0.60,
        'steps_per_cycle': 1*1e0,
        'train_per_cycle': 1,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        #'constraint': True,
        'constraint': False,
        'approx_value': True,
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None, #None
        'history_horizon': 0,  # 0 is using current state only
    },

}

VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]

VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]

RENDER = True
def get_env_from_name(name):

    from envs.Duffing_oscillator_sim import DOS as env
    env = env()
    env = env.unwrapped
  
    env.seed(SEED)
    return env

def get_train(name):
    if 'LAC' in name:
        from LAC.LAC_V1 import train

    return train

def get_policy(name):
    if 'LAC' in name :
        from LAC.LAC_V1 import LAC as build_func

    return build_func

def get_eval(name):
    if 'LAC' in name:
        from LAC.LAC_V1 import eval

    return eval


