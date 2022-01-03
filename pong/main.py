import re
import sys
import gym
import pylab
import numpy as np
from a2c_conv import A2CAgent
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# In case of CartPole-v1, maximum length of episode is 500
env = gym.make('Pong-v0')
# get size of state and action from environment
state_size = env.observation_space.shape
# print(f"state: {state_size}")
action_size = env.action_space.n
    
# make A2C agent
agent = A2CAgent(state_size, action_size)

scores, episodes = [], []
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)
for e in range(2):
    done = False
    score = 0
    state = env.reset()
    
    states = deque()
    states.extendleft([state])
    states.extendleft([state])
    #[print(x.shape) for x in queue]
    
    curr = 0
    
    while not done:
        # if agent.render:
        # env.render()

        
        allstates = tf.concat([x for x in states], 0)
        
        allstates = np.reshape(allstates, [1, 2, 210, 160, 3])


        action = agent.get_action(allstates)
        next_state, reward, done, info = env.step(action)

        if (reward == 1 or reward == -1):
            curr += 1
            print(f"Current score: {curr}")

        

        prev_states = tf.concat([x for x in states], 0)
        prev_states = np.reshape(prev_states, [1, 2, 210, 160, 3])
        states.pop()
        states.extendleft([next_state])

        allstates = tf.concat([x for x in states], 0)
        allstates = np.reshape(allstates, [1, 2, 210, 160, 3])
        # if an action make the episode end, then gives penalty of -100

        agent.train_model(prev_states, action, reward, allstates, done)

        score += reward
        state = next_state

        if done:
            # every episode, plot the play time
            scores.append(score)
            episodes.append(e)

            agent.actor.save_weights("./model/pong_actor.h5")
            agent.critic.save_weights("./model/pong_critic.h5")

            print("episode:", e, "  score:", score)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)
print(f"Episodes: {episodes}")
print(f"Scores: {scores}")

plt.plot(episodes, scores, label='Pong score')

plt.show()