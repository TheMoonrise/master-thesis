"""
Validation of an agent.
"""

import argparse
import json
import os
import re
import time
import torch

import numpy as np

from thesis.training.setup import Setup


def validate(trainer, config, checkpoint, episodes, trajectory_path):
    """
    Validates the training results by running on the environment and outputting the results.
    :param trainer: The trainer class to validate.
    :param config: The training and validation configuration.
    :param checkpoint: The path to the checkpoint to restore the model from.
    :param episodes: The number of episodes to validate for.
    :param trajectory_path: The path at which to store the trajectory.
    If this value is < 0 validation will continue indefinitely.
    """
    with torch.no_grad():
        # collect trajectory information
        trajectory = []

        # env = GridEnv(config['env_config'])
        env_config = config['env_config'] if 'env_config' in config else {}
        if 'is_validation' in env_config: env_config['is_validation'] = True

        sleep = env_config['timeout'] if 'timeout' in env_config else 0

        # check if rnn functionality is used
        is_rnn = 'model' in config and 'use_attention' in config['model'] and config['model']['use_attention']

        # restore the agent from the checkpoint
        agent = trainer(config=config)
        agent.restore(checkpoint)

        attention_dim = agent.get_policy().config['model']['attention_dim'] if is_rnn else 1
        state_rnn = [np.zeros((1, attention_dim))]

        env = agent.env_creator(env_config)
        state = env.reset()

        trajectory.append([list(state) + [0] if type(state) == np.ndarray else [state, 0]])

        tracker = (0, 0, 0)
        tracker_a = (0, 0, 0)  # tracking per action
        episode_reward = 0

        while episodes != 0:

            if is_rnn:
                action, state_out, _ = agent.compute_single_action(state, state_rnn, explore=False)
                state_rnn.append(state_out)

                # TODO gather the correct memory state size from the model view requirements
                state_rnn = state_rnn[-50:]
            else:
                action = agent.compute_single_action(state, explore=False)

            state, reward, done, _ = env.step(action)
            episode_reward += reward

            tracker_a = (tracker_a[0] + 1, tracker_a[1] + reward, tracker_a[2] + reward ** 2)

            env.render()
            trajectory[-1].append(list(state) + [reward] if type(state) == np.ndarray else [state, reward])
            if sleep > 0: time.sleep(sleep)

            if done:
                episodes -= 1
                if episodes != 0: state = env.reset()
                state_rnn = [np.zeros((1, attention_dim))]

                # update and print the current average performance
                tracker = (tracker[0] + 1, tracker[1] + episode_reward, tracker[2] + episode_reward ** 2)
                episode_reward = 0

                avg_reward = tracker[1] / tracker[0]
                avg_reward_a = tracker_a[1] / tracker_a[0]

                variance = tracker[2] / tracker[0] - avg_reward ** 2
                variance_a = tracker_a[2] / tracker_a[0] - avg_reward_a ** 2

                print((f'\rEpisode: {tracker[0]:0>3}'
                       f' Total Reward: {tracker[1]:<9.2f}'
                       f' Avg Reward: {avg_reward:<9.3f}'
                       f' Avg Reward [A]: {avg_reward_a:<9.3f}'
                       f' Variance: {variance:<9.5f}'
                       f' Variance [A]: {variance_a:<9.5f}'), end='')

                env.render()
                trajectory.append([])
                if sleep > 0: time.sleep(sleep)

        print(' ')
        env.close()

        # save the trajectory if a path is given
        if trajectory_path:
            with open(trajectory_path, 'w') as stream:
                stream.write(json.dumps(trajectory[:-1]))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('params', help='The configuration YAML file for the training parameters', type=str)
    parse.add_argument('checkpoint', help='The path to the checkpoint from which the model is loaded', type=str)

    parse.add_argument('--episodes', help='The number of episodes to validate for', type=int, default=-1)
    parse.add_argument('--debug', help='When set validation is single threaded', action='store_true')
    parse.add_argument('--trajectory', help='The path at which to store the trajectories', type=str)

    args = parse.parse_args()

    print('Beginning validation setup...')

    # run the setup for adding custom environments and initializing ray
    setup = Setup()
    setup.setup(args.debug)

    params = setup.parameters(args.params)

    config_path = os.path.join(os.path.dirname(args.checkpoint), '..', 'params.json')
    setup.update_hyperparameters(params, config_path)

    # fetch the trainer for the requested model
    trainer = setup.trainer(params['run'])

    # validate the results
    # validate(trainer, params['config'], args.checkpoint, args.episodes)
    validate(trainer, params['config'], args.checkpoint, args.episodes, args.trajectory)

    setup.shutdown()
    print('Validation completed')
