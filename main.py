from social_dilemmas.envs import env_creator
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
from social_dilemmas.envs.apple_learning import AppleLearningEnv
from social_dilemmas.envs.agent import TRAPPED_ACTIONS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Parameters from social influences paper
gamma = 0.99
n_agents = 1
n_actions = len(BASE_ACTIONS)  # amount of actions
#{0: 'MOVE_LEFT', 1: 'MOVE_RIGHT', 2: 'MOVE_UP', 3: 'MOVE_DOWN', 4: 'STAY', 5: 'TURN_CLOCKWISE', 6: 'TURN_COUNTERCLOCKWISE', 7: 'FIRE', 8: 'CLEAN'}
kernel_size = 3
strides=(1, 1)
n_output_conv = 6
n_dense_layers = 32
n_lstm = 128

env = AppleLearningEnv(num_agents=n_agents)
env.setup_agents()

agents = env.agents

state_size = None

A2C_agents = {}

for agentKey in agents.keys():
    A2C_agents[agentKey] = A2CAgent(state_size, n_actions, 0.0001, 0.0005) 
print("All agents(A2C)", agents)

#print(agents.items())
#dict_items([('agent-0', <social_dilemmas.envs.agent.CleanupAgent object at 0x7f8472108490>)])




from random import randrange
# randrange(10)


#print(n_steps)
# collective_reward = np.zeros(n_steps)


##----------------------------##
## Delete all previous images ##
##----------------------------##

import os
import glob

# files = glob.glob('/home/liguedino/Documents/github/project_comp_game_theory/images/*')
# for f in files:
    # print(f"Image: {f}")
    # os.remove(f)


##---------------##
## Training loop ##
##---------------##


totaller=0
n_episodes = 500
n_steps = int(100)
episode_rewards = np.zeros(n_episodes)
episode_length = np.zeros(n_episodes)
for episode in range(n_episodes):

    states = env.reset()

    agentQueus = {}

    for agentKey in A2C_agents.keys():
        state = states[agentKey]["curr_obs"]
        # state = np.reshape(state, [1, 15, 15, 3])
        #print(f"F {state.shape}")
        queue = deque()
        queue.extendleft([state])
        # queue.extendleft([state])
        #[print(x.shape) for x in queue]
        agentQueus[agentKey] = queue

    cum_rew = 0.
    stepDone = False

    for step in range(n_steps):
        if stepDone:
            break
        # if step == 0:
        #     env.render(f"images/{str(episode).zfill(10)}.png")
        curr_actions = {}
        for agentKey, agentObject in A2C_agents.items():
            queue = agentQueus[agentKey]
            allstates = tf.concat([x for x in queue], 0)
            allstates = np.reshape(allstates, [1, 1, 15, 15, 3])
            action = agentObject.get_action(allstates)
            curr_actions[agentKey] = action
        next_states, rewards, dones, _info = env.step(curr_actions) # agent id
        
        
        for agentKey, agentObject in A2C_agents.items():
            next_state = next_states[agentKey]["curr_obs"]

            reward = rewards[agentKey]
            cum_rew += reward

            if (reward==1):
                stepDone = True


            
            # collective_reward[step] += float(reward)
            done = dones[agentKey]
            action = curr_actions[agentKey]

            queue = agentQueus[agentKey]
            prev_states = tf.concat([x for x in queue], 0)
            prev_states = np.reshape(prev_states, [1, 1, 15, 15, 3])
            queue.pop()
            queue.extendleft([next_state])
            allstates = tf.concat([x for x in queue], 0)
            allstates = np.reshape(allstates, [1, 1, 15, 15, 3])

            agentObject.train_model(step, prev_states, action, reward, allstates, done)

    episode_length[episode] = step
    episode_rewards[episode] = cum_rew

    print(f"[{episode}] Episode rewards: {cum_rew} after {step+1} steps")
    # for agentKey, agentObject in A2C_agents.items():
    #     agentObject.save()   



d = {'Episodes': np.array(range(len(episode_rewards))), "AmountOfSteps": episode_length, 'Rewards': episode_rewards}
df = pd.DataFrame(d)
df.to_csv('trappedBoxTest.csv', index=False)

print(f"Finished 12.000 episodes")
print(f"All rewards: {episode_rewards}")


