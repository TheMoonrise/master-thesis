"""
Runs the given number of episode with baseline policies.
"""
import argparse
import os
import random
import json
import numpy as np

from crypto_markets_gym.envs.crypto_markets_env import CryptoMarketsEnv


def validate(episodes, action=0, random_action=False, no_validation=False, out=None):

    seed = 1214
    np.random.seed(seed)
    random.seed(seed)

    config = {}
    config['starting_funds'] = 100
    config['fee_difficulty_scaling_factor'] = 0
    config['test_data_usage_percentage'] = 0.2
    config['is_validation'] = not no_validation

    env = CryptoMarketsEnv(config)
    state = env.reset()

    tracker = (0, 0, 0)
    tracker_a = (0, 0, 0)

    episode_reward = 0
    episode_list = []

    while episodes != 0:

        if random_action: action = np.random.choice(11)

        state, reward, done, _ = env.step(np.array([action]))
        tracker_a = (tracker_a[0] + 1, tracker_a[1] + reward, tracker_a[2] + reward ** 2)
        episode_reward += reward

        if done:
            episodes -= 1
            if episodes != 0: state = env.reset()

            # update and print the current average performance
            tracker = (tracker[0] + 1, tracker[1] + episode_reward, tracker[2] + episode_reward ** 2)

            episode_list.append(episode_reward)
            episode_reward = 0

            avg_reward = tracker[1] / tracker[0]
            avg_reward_a = tracker_a[1] / tracker_a[0]

            variance = tracker[2] / tracker[0] - avg_reward ** 2
            variance_a = tracker_a[2] / tracker_a[0] - avg_reward_a ** 2

            print((f'\rEpisode: {tracker[0]:0>3}'
                   f' Total Reward: {tracker[1]:<9.2f}'
                   f' Avg Reward: {avg_reward:<9.7f}'
                   f' Avg Reward [A]: {avg_reward_a:<9.7f}'
                   f' Var: {variance:<9.7f}'
                   f' Var [A]: {variance_a:<9.7f}'), end='')

            env.render()

    if out:
        file_name = 'bhs'
        file_name += '_noval' if no_validation else '_val'
        file_name += '_rand' if random_action else '_' + str(action)

        with open(os.path.join(out, file_name + '.json'), 'w') as stream:
            stream.write(json.dumps(episode_list))

    print(' ')
    env.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--episodes', help='The number of episodes to validate for', type=int, default=-1)
    parse.add_argument('--output-dir', help='Path to store results at', type=str)
    parse.add_argument('--actions', help='The comma separated list of actions to validate', type=str, default="0")
    parse.add_argument('--random', help='When set random actions are used', action='store_true')
    parse.add_argument('--no-validation', help='When set the training data is used', action='store_true')

    args = parse.parse_args()

    for a in args.actions.split(','):
        try:
            action = int(a)
            print('\na =', action)
            validate(args.episodes, action, args.random, args.no_validation, args.output_dir)
        except Exception as e:
            print('Illegal action:', a)
            print(e)

    print('Validation completed')
