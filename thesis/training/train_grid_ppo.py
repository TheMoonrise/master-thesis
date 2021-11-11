"""
Training of a PPO agent on the grid world
"""

import argparse
import os

from thesis.environments.grid import GridEnv


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('path', help='The configuration JSON file for the grid world generation')
    args = parse.parse_args()

    # transform the relative grid path to a absolute path
    grid_path = os.path.join(os.getcwd(), args.path)

    # create the grid environment
    env = GridEnv(grid_path)

    # perform training
    obs = env.reset()
