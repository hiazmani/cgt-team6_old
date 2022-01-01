from social_dilemmas.envs.cleanup import CleanupEnv
import matplotlib.pyplot as plt
import numpy as np

class agent():
    def __init__(self) -> None:
        pass

env = CleanupEnv()
env.setup_agents()

agents = env.agents

init_obs = env.reset()

print(init_obs)