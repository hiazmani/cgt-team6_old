from social_dilemmas.envs import agent
from social_dilemmas.envs.agent import FreeAgent
from social_dilemmas.maps import TRAPPED_BOX_MAP2

from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.envs.agent import AppleAgent
from numpy.random import rand

import numpy as np

_APPLE_LEARNING_AGENT_ACTS = {}

APPLE_LEARNING_VIEW_SIZE = 7

class AppleLearningEnv(MapEnv):
    def __init__(
        self,
        ascii_map=TRAPPED_BOX_MAP2,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,                   
    ):
        super().__init__(
            ascii_map,
            _APPLE_LEARNING_AGENT_ACTS,
            APPLE_LEARNING_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.apple_points.append([row, col])
                # if self.base_map[row, col] == b"B":
                #     self.apple_points.append([row, col])

    @property
    def action_space(self):
        return DiscreteWithDType(8, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        agent_id = "agent-" + str(0)
        points = self.spawn_points
        random = np.random.randint(len(self.spawn_points))
        spawn_point = points[random]
        
        rotation = self.spawn_rotation()
        grid = map_with_agents
        agent = AppleAgent(agent_id, spawn_point, rotation, grid, view_len=APPLE_LEARNING_VIEW_SIZE)
        self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        agent_positions = self.agent_pos

        random = np.random.randint(len(self.apple_points))
        pos = self.apple_points[random]
        while pos in agent_positions:
            random = np.random.randint(len(self.apple_points))
            pos = self.apple_points[random]
        self.single_update_map(pos[0], pos[1], b"A")

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean

        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken

        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        updates = []        
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """
        new_apple_points = []

        n_apples = 0

        agent_positions = self.agent_pos

        # Check whether there already is an apple somewhere
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # Check whether this location contains an apple
            if self.world_map[row, col] == b"A":
                return new_apple_points
                # n_apples += 1
            # if n_apples == 2:
                # There is already an apple somewhere
                # return new_apple_points
        # Place an apple
        for i in range(len(self.apple_points)):
            if n_apples == 0:
                return new_apple_points
            random = np.random.randint(len(self.apple_points))
            row, col = self.apple_points[random]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                new_apple_points.append((row, col, b"A"))
                # n_apples -= 1
                return new_apple_points
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples