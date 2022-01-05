# from social_dilemmas.envs import env_creator
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv
from social_dilemmas.envs.agent import BASE_ACTIONS, CLEANUP_ACTIONS, HARVEST_ACTIONS, TRAPPED_ACTIONS, APPLE_ACTIONS
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from a2c_conv import *
import pandas as pd

from collections import deque

from social_dilemmas.envs.trapped_box import TrappedBoxEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.agent import HARVEST_ACTIONS
from social_dilemmas.envs.apple_learning import AppleLearningEnv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Parameters from social influences paper
gamma = 0.99
n_agents = 2
n_actions = len(HARVEST_ACTIONS)  # amount of actions
#{0: 'MOVE_LEFT', 1: 'MOVE_RIGHT', 2: 'MOVE_UP', 3: 'MOVE_DOWN', 4: 'STAY', 5: 'TURN_CLOCKWISE', 6: 'TURN_COUNTERCLOCKWISE', 7: 'FIRE', 8: 'CLEAN'}
kernel_size = 3
strides=(1, 1)
n_output_conv = 6
n_dense_layers = 32
n_lstm = 128

env = TrappedBoxEnv(num_agents=n_agents)
env.setup_agents()

def print_kleene(kleene):
    for x in range(len(kleene)):
        for y in range(len(kleene[0])):
            if (kleene[x][y] == [0,0,0]).all():
                print(" ", end='')
            elif (kleene[x][y] == [0,0,255]).all():
                print("K", end='')
            elif (kleene[x][y] == [180, 180, 180]).all():
                print("M", end='')
            elif (kleene[x][y] == [0, 255, 0]).all():
                print("A", end='')
            else:
                print("P", end='')
        print("")

agents = env.agents


obs = env.reset()
env.render("test.png")

for i in range(1):
    env.reset()
    
    print("ACTUAL")
    obs, rewards, dones, infos = env.step({"agent-1": 2})
    kleene = obs["agent-1"]["curr_obs"]
    print_kleene(kleene)
    print("\n\n")

    print("DUMMY")
    obs, rewards, dones, infos = env.dummy_step({"agent-1": 2})
    kleene = obs["agent-1"]["curr_obs"]
    print_kleene(kleene)
    print("\n\n")

    print("ACTUAL")
    obs, rewards, dones, infos = env.step({"agent-1": 4})
    kleene = obs["agent-1"]["curr_obs"]
    print_kleene(kleene)
    print("\n\n")
 



    

    env.render(f"test-{i}.png")

# print(kleene)

# for x in range(len(kleene)):
#     for y in range(len(kleene[0])):
#         print(kleene[x][y])

# print()

