"""
Validation of an agent.
"""

import argparse
import time
import torch

from thesis.training.setup import Setup


def validate(trainer, config, checkpoint):
    """
    Validates the training results by running on the environment and outputting the results.
    :param trainer: The trainer class to validate.
    :param config: The training and validation configuration.
    :param checkpoint: The path to the checkpoint to restore the model from.
    """
    with torch.no_grad():
        # restore the agent from the checkpoint
        agent = trainer(config=config)
        agent.restore(checkpoint)

        # env = GridEnv(config['env_config'])
        env_config = config['env_config'] if 'env_config' in config else {}
        sleep = env_config['timeout'] if 'timeout' in env_config else 0

        env = agent.env_creator(env_config)
        sta = env.reset()

        tracker = (0, 0, 0)
        episode_reward = 0

        while True:

            act = agent.compute_single_action(sta, explore=False)
            sta, reward, done, _ = env.step(act)
            episode_reward += reward

            env.render()
            if sleep > 0: time.sleep(sleep)

            if done:
                sta = env.reset()

                # update and print the current average performance
                tracker = (tracker[0] + 1, tracker[1] + episode_reward, tracker[2] + episode_reward ** 2)
                episode_reward = 0

                avg_reward = tracker[1] / tracker[0]
                variance = tracker[2] / tracker[0] - avg_reward ** 2

                print(f'Avg Reward: {avg_reward:<5.0f} Variance: {variance:<5.0f}', end='\r')

                env.render()
                if sleep > 0: time.sleep(sleep)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('params', help='The configuration YAML file for the training parameters', type=str)
    parse.add_argument('checkpoint', help='The path to the checkpoint from which the model is loaded', type=str)

    args = parse.parse_args()

    print('Beginning validation setup...')

    # run the setup for adding custom environments and initializing ray
    setup = Setup()
    setup.setup()

    params = setup.parameters(args.params)

    # fetch the trainer for the requested model
    trainer = setup.trainer(params['run'])

    # validate the results
    validate(trainer, params['config'], args.checkpoint)
