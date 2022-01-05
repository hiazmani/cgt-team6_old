import re
import sys
import gym
import pylab
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
import os


# gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tensorflow.config.experimental.set_memory_growth(device, True)


##https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py

EPISODES = 1000


# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, observation_size, action_size, actor_lr, critic_lr, gamma, saved_weights_dir = None):
        # get size of state and action
        self.observation_size = observation_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = gamma
        self.actor_lr = actor_lr #0.001
        self.critic_lr = critic_lr #0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # if saved_weights_dir and os.listdir(saved_weights_dir) != []:
        #     self.actor.load_weights(f"./{saved_weights_dir}/a2c_actor.h5")
        #     self.critic.load_weights(f"./{saved_weights_dir}/a2c_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Conv2D(6, (3, 3), activation='linear', strides=(1, 1), input_shape=(1, 15, 15, 3)))
        actor.add(Flatten())
        actor.add(Dense(32, activation='linear', kernel_initializer='he_uniform'))
        actor.add(Dense(32, activation='linear', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation="softmax"))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Conv2D(6, (3, 3), activation='linear', strides=(1, 1), input_shape=(1, 15, 15, 3)))
        critic.add(Flatten())
        critic.add(Dense(32, activation='linear', kernel_initializer='he_uniform'))
        critic.add(Dense(32, activation='linear', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation="softmax"))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        # Predict the policy (probability of each action)
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # using the output of policy network, return probabilities of actions
    def get_action_probabilities(self, state):
        return self.actor.predict(state, batch_size=1).flatten()

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    # def save(self):
    #     self.actor.save_weights("./model/a2c_actor.h5")
    #     self.critic.save_weights("./model/a2c_critic.h5")