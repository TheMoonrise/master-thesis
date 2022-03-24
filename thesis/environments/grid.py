"""
Grid world implementation following OpenAI standards.
"""

import os
import json

from collections import defaultdict

import time
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt


class GridEnv(gym.Env):
    """
    Grid environment providing an interface for agent interaction.
    """

    def __init__(self, config, setup=None):
        """
        Loads the environment configuration.
        :param config: The config dict provided through the RlLib trainer.
        :param setup: Optional json string replacing the grid configuration loaded from disc.
        """
        self.origins = []
        self.targets = []
        self.actions = defaultdict(lambda: None)

        self.position = [0, 0]
        self.done = True

        self.figure = None
        self.axes = None

        # read the configuration from the provided file
        # file is not checked for validity
        if not setup:
            with open(os.path.join(config['root'], config['grid_path'])) as file:
                setup = file.read()

        setup = json.loads(setup.lower())

        # the world is defined as a list of strings
        # each string represents a row of the environment from left to right
        # the first string represents the lower most row of the environment
        self.world = np.zeros((len(setup['grid']), len(setup['grid'][0]), 2))

        for r, s in enumerate(setup['grid']):
            for c, t in enumerate(s):
                self.world[r, c] = setup[t]

                # record origin and target states
                if t in setup['origin']: self.origins.append((r, c))
                if t in setup['target']: self.targets.append((r, c))

        # define action and state spaces
        # divide world size by two because each tile has two value entries
        self.observation_space = gym.spaces.Discrete(self.world.size // 2)

        # when meta actions are enabled the action sampling is done in the environment
        self.meta_actions = 'meta_actions' in config and config['meta_actions']

        if self.meta_actions: self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        else: self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        """
        Advances the environment by one step.
        :param action: The action to be executed.
        Actions can take the values 0, 1, 2, 3 corresponding to the directions UP, RIGHT, DOWN, LEFT.
        :return: The new state the environment is now in.
        :return: The reward received by the agent for the action taken.
        :return: Whether a terminal state is reached.
        :return: And info object providing additional information on the current state.
        """
        if self.done: print('The environment is in a terminal state. Reset the environment before stepping')

        # sample an actual action if meta actions are enabled
        if self.meta_actions:
            # store the action distribution for the current state
            current_state_number = self.state_number(self.position)
            self.actions[str(current_state_number)] = action
            action = np.random.choice(4, p=action)

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
        self.position = np.array(self.origins[np.random.choice(len(self.origins))])
        self.done = False
        return self.state_number(self.position)

    def render(self, mode='human'):
        """
        Renders the current grid environment using pyplot.
        Each iteration introduces a brief delay. Therefore this functions should be avoided for training.
        :param model: Rendering mode for open ai compatability.
        """
        # if the render function is called for the first time
        # the figure is configured
        if not self.figure:
            plt.rcParams['toolbar'] = 'None'

            self.figure = plt.figure(facecolor='white', figsize=np.array(self.world.shape[:2]) * 1.2)

            self.axes = plt.Axes(self.figure, [0, 0, 1, 1])
            self.figure.add_axes(self.axes)

            plt.get_current_fig_manager().set_window_title('Grid Environment')
            plt.ion()
            plt.show()

        self.axes.clear()
        self.axes.set_axis_off()

        self.axes.hlines(np.arange(0.5, self.world.shape[0] - 1), -0.5, self.world.shape[1], color='k', linewidth=4)
        self.axes.vlines(np.arange(0.5, self.world.shape[1] - 1), -0.5, self.world.shape[0], color='k', linewidth=4)

        # build a matrix to display the grid environment by
        grid = np.zeros(self.world.shape[:2])
        for t in self.origins: grid[t] = 4
        for t in self.targets: grid[t] = 6
        grid[tuple(self.position)] = 10

        # insert reward signals into the graphics
        for r, c in np.ndindex(grid.shape):
            text = f'N({self.world[r, c, 0]}, {self.world[r, c, 1]})'
            self.axes.text(c, r + .18, text, ha='center', va='center', fontsize='x-small',
                           weight='bold', family='monospace', color='white')

            for action in range(4):
                state_number = str(self.state_number((r, c)))
                if self.actions[state_number] is None: break
                value = self.actions[state_number][action]
                text = str(round(value, 2))

                offset = .35
                delta_c = (2 - action) * (action % 2) * offset
                delta_r = (1 - action) * ((action + 1) % 2) * offset
                self.axes.text(c + delta_c, r + delta_r, text, ha='center', va='center',
                               fontsize='x-small', color='white', family='monospace', weight='bold')

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
        return position[0] * self.world.shape[1] + position[1]


if __name__ == '__main__':
    # Perform random moves and rendering for debug purposes
    parse = argparse.ArgumentParser()
    parse.add_argument('path', help='The path ot the grid config to debug', type=str)
    args = parse.parse_args()

    config = {'root': os.getcwd(), 'grid_path': args.path}
    grid = GridEnv(config)
    done = True

    while True:
        if done:
            grid.reset()
            done = False
        else:
            _, _, done, _ = grid.step(np.random.choice(4))

        grid.render()
        time.sleep(.2)
