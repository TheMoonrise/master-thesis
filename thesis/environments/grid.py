"""
Grid world implementation following OpenAI standards
"""

import json
import numpy as np


class GridEnv():
    """
    Grid environment providing an interface for agent interaction
    """

    def __init__(self, grid_path):
        """
        Loads the environment configuration.
        :param grid_path: The file path for the JSON file containing the environment definition
        """
        self.origins = []
        self.targets = []

        self.position = [0, 0]
        self.done = True

        # read the configuration from the provided file
        # file is not checked for validity
        with open(grid_path) as file:
            setup = json.loads(file.read().lower())

        # the world is defined as a list of strings
        # each string represents a row of the environment from left to right
        # the first string represents the lower most row of the environment
        self.world = np.zeros((len(setup['grid'][0]), len(setup['grid']), 2))

        for y, s in enumerate(setup['grid']):
            for x, c in enumerate(s):
                self.world[x, y] = setup[c]

                # record origin and target states
                if c == setup['origin']: self.origins.append((x, y))
                if c == setup['target']: self.targets.append((x, y))

    def step(self, action):
        """
        Advances the environment by one step
        :param action: The action to be executed.
        Actions can take the values 0, 1, 2, 3 corresponding to the directions UP, RIGHT, DOWN, LEFT
        :return: The new state the environment is now in
        :return: The reward received by the agend for the action taken
        :return: Whether a terminal state is reached
        :return: And info object providing additional information on the current state
        """
        if self.done: print('The environment is in a terminal state. Reset the environment before stepping')

        delta_x = (2 - action) * (action % 2)
        delta_y = (1 - action) * ((action + 1) % 2)

        self.position = np.add(self.position, (delta_x, delta_y))
        self.position = np.clip(self.position, (0, 0), self.world.shape[:2])

        nrm = self.world[tuple(self.position)]
        reward = np.random.normal(nrm[0], 0 if nrm[1] == 0 else np.sqrt(nrm[1]))

        if tuple(self.position) in self.targets: self.done = True
        return self.state_number(self.position), reward, self.done, None

    def reset(self):
        """
        Return the environment into the initial state
        :return: The initial state
        """
        self.position = self.origins[np.random.choice(len(self.position))]
        self.done = False
        return self.state_number(self.position)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def state_number(self, position):
        """
        Converts a position tuple into a unique state number
        :param position: The current position as tuple (x, y)
        :return: The position as a single int
        """
        return position[0] * self.world.shape[0] + position[1]
