"""

The present code is developed for the paper entitled: "Safe Reinforcement Learning-Based Vibration Control: Overcoming Training Risks with LQR Guidance"

It is important to note that the implementation of the LAC algorithm is based on, and adapted from, the work of: Han, Minghao, Lixian Zhang, Jun Wang, and Wei Pan. “Actor-critic reinforcement
learning for control with stability guarantee.” IEEE Robotics and Automation Let-
ters 5, no. 4 (2020): 6217-6224.

"""

import tensorflow as tf
import os
from variant import VARIANT, get_env_from_name,  get_train, get_eval
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    root_dir = VARIANT['log_path']
    if VARIANT['train']:
        for i in range(VARIANT['start_of_trial'], VARIANT['start_of_trial']+VARIANT['num_of_trials']):
            VARIANT['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + VARIANT['log_path'])
            train = get_train(VARIANT['algorithm_name'])
            train(VARIANT)

            tf.reset_default_graph()
            
    else:
        eval = get_eval(VARIANT['algorithm_name'])
        eval(VARIANT)
        
        