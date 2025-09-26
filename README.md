[Note: Add animated plots, personal website for paper, add citation]

# LQR-Guided-RL-controller
This repository contains the code used to produce the results presented in the ICCMS 2025 conference paper titled **"Safe Reinforcement Learning-Based Vibration Control: Overcoming Training Risks with LQR Guidance"**.

## Overview
This project implements a reinforcement learning (RL) based vibration control system guided by Linear Quadratic Regulator (LQR) principles. The main objective is to achieve safe and effective control in vibration systems by integrating classical LQR control insights into the RL policy training process.

## Features
* Implementation of LQR-Guided Reinforcement Learning policy.

* Simulation setup and environment for vibration control systems.

* Comparative analysis with baseline controllers.

* Tools for training, evaluation, and visualization of control policies.

## Directory Description
* Generate plots/: Contains scripts to generate all the figures used in the paper.

* LQR-Guided RL code/: Main source directory with training and testing scripts for the LQR-guided reinforcement learning controller.

* LQR-Guided RL code/main.py: Primary execution script for initiating either training or testing procedures.

* LQR-Guided RL code/variant.py: Module specifying the hyperparameter configurations used in the experiments.

* LQR-Guided RL code/data/: Directory containing datasets and input files necessary for execution.

* LQR-Guided RL code/log/: Repository of trained model checkpoints and related outputs.
 
* LQR-Guided RL code/plots/: Directory storing the results obtained from testing.


## Citation
If you use this repository or code in your research, please cite our paper:
