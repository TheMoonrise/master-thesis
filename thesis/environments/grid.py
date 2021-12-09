"""
Grid world implementation following OpenAI standards.
"""

import os
import json
import gym
import numpy as np
import matplotlib.pyplot as plt


class GridEnv(gym.Env):
    """
    Grid environment providing an interface for agent interaction.
    """

    def __init__(self, config):
        """
        Loads the environment configuration.
        :param config: The config dict provided throught the RLlib trainer.
        """
        self.origins = []
        self.targets = []

        self.position = [0, 0]
        self.done = True

        self.figure = None
        self.axes = None

        # read the configuration from the provided file
        # file is not checked for validity
        with open(os.path.join(config['root'], config['grid_path'])) as file:
            setup = json.loads(file.read().lower())

        # the world is defined as a list of strings
        # each string represents a row of the environment from left to right
        # the first string represents the lower most row of the environment
        self.world = np.zeros((len(setup['grid']), len(setup['grid'][0]), 2))

        for r, s in enumerate(setup['grid']):
            for c, t in enumerate(s):
                self.world[r, c] = setup[t]

                # record origin and target states
                if t == setup['origin']: self.origins.append((r, c))
                if t == setup['target']: self.targets.append((r, c))

        # define action and state spaces
        # divide world size by two because each tile has two value entries
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(self.world.size // 2)

    def step(self, action):
        """
        Advances the environment by one step.
        :param action: The action to be executed.
        Actions can take the values 0, 1, 2, 3 corresponding to the directions UP, RIGHT, DOWN, LEFT.
        :return: The new state the environment is now in.
        :return: The reward received by the agend for the action taken.
        :return: Whether a terminal state is reached.
        :return: And info object providing additional information on the current state.
        """
        if self.done: print('The environment is in a terminal state. Reset the environment before stepping')

        delta_c = (2 - action) * (action % 2)
        delta_r = (1 - action) * ((action + 1) % 2)

        self.position = np.add(self.position, (delta_r, delta_c))
        self.position = np.clip(self.position, (0, 0), np.subtract(self.world.shape[:2], 1))

        normal = self.world[tuple(self.position)]
        reward = np.random.normal(normal[0], 0 if normal[1] == 0 else np.sqrt(normal[1]))

        if tuple(self.position) in self.targets: self.done = True
        return self.state_number(self.position), reward, self.done, {}

    def reset(self):
        """
        Return the environment into the initial state
        :return: The initial state
        """
        self.position = np.array(self.origins[np.random.choice(len(self.position))])
        self.done = False
        return self.state_number(self.position)

    def render(self, mode='human'):
        """
        Renders the current grid environment using pyplot.
        Each iteration introduces a brief dely. Therefore this functions should be avoided for training.
        :param model: Rendering mode for open ai compatability.
        """
        # if the render function is called for the first time
        # the figure is configured
        if not self.figure:
            plt.rcParams['toolbar'] = 'None'

            self.figure = plt.figure(facecolor='black')
            self.axes = self.figure.gca()
            self.axes.axis('off')

            plt.get_current_fig_manager().set_window_title('Grid Environment')
            plt.ion()
            plt.show()

        # build a matrix to display the grid environment by
        grid = np.ones(self.world.shape[:2])
        for t in self.origins: grid[t] = 0
        for t in self.targets: grid[t] = 3
        grid[tuple(self.position)] = 2

        # insert reward signals into the graphics
        for r, c in np.ndindex(grid.shape):
            text = f'N({self.world[r, c, 0]}, {self.world[r, c, 1]})'
            self.axes.text(c, r, text, ha='center', va='center')

        self.axes.matshow(grid)

        plt.draw()
        plt.pause(0.1)

    def close(self):
        pass

    def state_number(self, position):
        """
        Converts a position tuple into a unique state number.
        :param position: The current position as tuple (row, column).
        :return: The position as a single int.
        """
        return position[0] * self.world.shape[0] + position[1]
