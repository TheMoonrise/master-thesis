"""
Implementation of a crypto trading environment.
"""

import os
import math
from io import StringIO

import gym
import pandas as pd
import numpy as np


class MarketEnv(gym.Env):
    """
    Market environment providing an interface for agent interaction.
    """
    data_cache = {}

    def __init__(self, config, setup=None):
        """
        Loads the environment configuration.
        :param config: The config dict provided through the RlLib trainer.
        :param setup: Optional data string replacing the grid configuration loaded from disc.
        """
        self.invest_index = 0
        self.done = True

        self.episode_start = 0
        self.episode_count = 0
        self.episode_progress = 0

        self.transaction_fee = 0.002 if 'transaction_fee' not in config else config['transaction_fee']
        if self.transaction_fee > 0: self.transaction_fee = math.log(1 - self.transaction_fee)

        assert 'episode_length' in config
        self.episode_length = config['episode_length']
        self.validation_step = self.episode_length if 'validation_step' not in config else config['validation_step']

        self.is_training = 'is_validation' not in config or not config['is_validation']

        self.setup_environment_data(config, setup)

        # the observation space includes the values for all assets as well as the currently active asset
        self.coin_count = len(self.coin_labels)
        self.observation_space = gym.spaces.Box(low=0, high=100_000, shape=(self.coin_count + 1,))

        # when meta actions are enabled the action sampling is done in the environment
        self.meta_actions = 'meta_actions' in config and config['meta_actions']

        if self.meta_actions: self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.coin_count,))
        else: self.action_space = gym.spaces.Discrete(self.coin_count)

        print('\n\nMARKET SETUP\n')

    def setup_environment_data(self, config, setup):
        """
        Loads and processes the data for the environment.
        :param config: The config dict provided through the RlLib trainer.
        :param setup: Optional data string replacing the grid configuration loaded from disc.
        """
        if not setup:
            assert 'data_path' in config
            assert 'root' in config

        if setup or config['data_path'] not in MarketEnv.data_cache:
            data_string = setup

            # read the data csv file from disk if no market data was provided as string
            if not setup:
                data_path = config['data_path'].replace('\\', '/')
                with open(os.path.join(config['root'], data_path)) as file:
                    data_string = file.read()

            # create a dataframe from the history crypto data
            df = pd.read_csv(StringIO(data_string))
            self.coin_labels = ['STABLE'] + list(df.columns)[1:]

            data = df.to_numpy()[:, 1:]
            # insert a stable coin into the data matrix after the timestamp
            data = np.insert(data, 0, 1, axis=1)

            if not setup:
                MarketEnv.data_cache[config['data_path']] = (data, self.coin_labels)

        else:
            # use cached market data to avoid loading multiple times
            data = MarketEnv.data_cache[config['data_path']][0]
            self.coin_labels = MarketEnv.data_cache[config['data_path']][1]

        self.data = data

        # create a dataframe to store trajectories for validation
        self.trajectory = pd.DataFrame(columns=self.coin_labels + ['INVESTMENT', 'RETURN'], dtype=float)
        self.trajectory_output = config['validation_output'].replace('\\', '/') if 'validation_output' in config else None

        # split the data into training and validation
        validation_length = 0 if 'validation_length' not in config else config['validation_length']

        if validation_length > 0:
            self.data = self.data[:-validation_length] if self.is_training else self.data[-validation_length:]

    def step(self, action):
        """
        Advances the environment by one step.
        :param action: The action to be executed.
        When working with regular actions the action defines the index of the asset to interact with.
        If the selected asset is already selected it is held.
        If a different asset is selected the current asset is sold and the new asset is purchased.
        When working with meta actions the actions define the probability to interact with each asset.
        :return: The new state the environment is now in.
        The state includes the current values for each asset as well as the currently held asset.
        :return: The reward received by the agent for the action taken.
        This is the log return from the asset that was held from the previous to the current state.
        :return: Whether a terminal state is reached.
        :return: And info object providing additional information on the current state.
        """
        if self.done:
            print('The environment is in a terminal state. Reset the environment before stepping')
            return np.zeros((self.coin_count + 1,)), 0, self.done, {}

        # check if the end of the episode is reached
        self.episode_progress += 1
        if (self.episode_progress >= self.episode_length - 1): self.done = True

        state_index = min(self.episode_start + self.episode_progress, self.data.shape[0])

        last_state = self.data[state_index - 1]
        curr_state = self.data[state_index]

        # sample an action if using meta actions
        if self.meta_actions: action = np.random.choice(self.coin_count, p=action)

        transaction_fee = 0 if action == self.invest_index else self.transaction_fee
        self.invest_index = action

        # determine the log return going to the current state
        tot_return = curr_state[self.invest_index] / last_state[self.invest_index]
        log_return = math.log(tot_return) if tot_return > 0 else 0

        # aggregate information in the trajectory
        if not self.is_training:
            row = pd.DataFrame([list(last_state) + [self.invest_index, log_return]], columns=self.trajectory.columns)
            self.trajectory = pd.concat((self.trajectory, row), ignore_index=True)

        return np.concatenate((curr_state, [self.invest_index])), log_return + transaction_fee, self.done, {}

    def reset(self):
        """
        Return the environment into the initial state.
        For this the next episode in the sequence of episodes is sampled.
        :return: The initial state.
        """
        self.episode_start = np.random.choice(max(self.data.shape[0] - self.episode_length, 1))

        if not self.is_training:
            self.episode_start = self.episode_count * self.validation_step

            if self.episode_start + self.episode_length >= self.data.shape[0]:
                print('Validation is exceeding the validation dataset size')

        self.episode_count += 1

        self.episode_progress = 0
        self.invest_index = 0
        self.done = False

        state = self.data[self.episode_start]
        return np.concatenate((state, [self.invest_index]))

    def render(self, mode='human'):
        """
        Renders the current grid environment using pyplot.
        Each iteration introduces a brief delay. Therefore this functions should be avoided for training.
        :param model: Rendering mode for open ai compatability.
        """
        pass

    def close(self):
        if self.is_training or not self.trajectory_output: return
        self.trajectory.to_csv(self.trajectory_output, index=False)
        print('Environment trajectory saved to', self.trajectory_output)
