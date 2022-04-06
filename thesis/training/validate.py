"""
Validation of an agent.
"""

import argparse
from random import Random
import time
import torch

import numpy as np

from thesis.training.setup import Setup


def validate(trainer, config, checkpoint, episodes):
    """
    Validates the training results by running on the environment and outputting the results.
    :param trainer: The trainer class to validate.
    :param config: The training and validation configuration.
    :param checkpoint: The path to the checkpoint to restore the model from.
    :param episodes: The number of episodes to validate for.
    If this value is <0 validation will continue indefinitely.
    """
    with torch.no_grad():
        # restore the agent from the checkpoint
        agent = trainer(config=config)
        agent.restore(checkpoint)

        # env = GridEnv(config['env_config'])
        env_config = config['env_config'] if 'env_config' in config else {}
        if 'is_validation' in env_config: env_config['is_validation'] = True

        sleep = env_config['timeout'] if 'timeout' in env_config else 0

        # check if rnn functionality is used
        is_rnn = 'model' in config and 'use_attention' in config['model'] and config['model']['use_attention']

        attention_dim = agent.get_policy().config['model']['attention_dim'] if is_rnn else 1
        state_rnn = [np.zeros((1, attention_dim))]

        env = agent.env_creator(env_config)
        state = env.reset()

        tracker = (0, 0, 0)
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

            env.render()
            if sleep > 0: time.sleep(sleep)

            if done:
                state = env.reset()
                state_rnn = [np.zeros((1, attention_dim))]

                # update and print the current average performance
                tracker = (tracker[0] + 1, tracker[1] + episode_reward, tracker[2] + episode_reward ** 2)
                episode_reward = 0
                episodes -= 1

                avg_reward = tracker[1] / tracker[0]
                variance = tracker[2] / tracker[0] - avg_reward ** 2

                print((f'\rEpisode: {tracker[0]:0>3}'
                       f' Total Reward: {tracker[1]:<9.2f}'
                       f' Avg Reward: {avg_reward:<9.3f}'
                       f' Variance: {variance:<9.3f}'), end='')

                env.render()
                if sleep > 0: time.sleep(sleep)

        print(' ')
        env.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('params', help='The configuration YAML file for the training parameters', type=str)
    parse.add_argument('checkpoint', help='The path to the checkpoint from which the model is loaded', type=str)

    parse.add_argument('--episodes', help='The number of episodes to validate for', type=int, default=-1)
    parse.add_argument('--debug', help='When set validation is single threaded', action='store_true')

    args = parse.parse_args()

    print('Beginning validation setup...')

    # run the setup for adding custom environments and initializing ray
    setup = Setup()
    setup.setup(args.debug)

    params = setup.parameters(args.params)

    # fetch the trainer for the requested model
    trainer = setup.trainer(params['run'])

    # validate the results
    validate(trainer, params['config'], args.checkpoint, args.episodes)

    setup.shutdown()
    print('Validation completed')
