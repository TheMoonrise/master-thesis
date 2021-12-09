"""
Training of an agent.
"""

import argparse
import os
import time
import yaml
import torch

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer


from thesis.environments.grid import GridEnv
from thesis.policies.ppo_risk_averse import risk_averse_trainer


def setup():
    """
    Prepares ray, and other components for training and validation.
    Register custom models with the rllib library.
    """
    ray.init()
    register_env('gridworld', GridEnv)


def selected_config(path):
    """
    Reads the config contents from the yaml file.
    :param path: The path to the yaml config file.
    :return: The config dictionary.
    """
    with open(path) as file:
        return yaml.safe_load(file)


def selected_trainer(model):
    """
    Provides the trainer class.
    :param model: The name of the model to train.
    :return: The trainer class for the input parameters. Default return is vanilla ppo.
    """
    if model == 'risk': return risk_averse_trainer()
    return PPOTrainer


def training(trainer, config, checkpoint_path, name, resume, iterations):
    """
    Performs a training run using tune.
    :param trainer: The trainer class to be used for the training.
    :param config: The configuration for the training.
    :param checkpoint_path: The path where to save the checkpoints to.
    :param name: The name of the training run.
    :param resume: Whether a previous run with the same name should be resumed.
    :param iterations: The number of iterations to run for.
    :return: The analyis containing training results.
    """
    stop = {'training_iteration': iterations}

    # run the training using tune for hyperparameter tuning
    # training results are stored in analysis
    analysis = tune.run(trainer, config=config, stop=stop, checkpoint_at_end=True,
                        local_dir=checkpoint_path, resume=resume, name=name)

    # return the analysis object
    return analysis


def validate(analysis, trainer, config):
    """
    Validates the training results by running on the environment and outputting the results.
    :param analysis: The analysis containing the training results.
    :param trainer: The trainer class to validate.
    :param config: The training and validation configuration.
    """
    with torch.no_grad():
        # perform some verification runs on the environment
        trial = analysis.get_best_logdir('episode_reward_mean', 'max')
        ckpnt = analysis.get_best_checkpoint(trial, 'training_iteration', 'max')

        # restore the agent from the checkpoint
        agent = trainer(config=config)
        agent.restore(ckpnt)

        # env = GridEnv(config['env_config'])
        env = agent.env_creator(config['env_config'])
        sta = env.reset()

        while True:
            act = agent.compute_single_action(sta)
            sta, _, done, _ = env.step(act)
            env.render()
            time.sleep(.5)

            if done:
                sta = env.reset()
                env.render()
                time.sleep(.5)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('path', help='The configuration YAML file for the training parameters', type=str)
    parse.add_argument('--root', help='The root path of the project. If none the current working directory is used', type=str)

    parse.add_argument('--model', help='The name of the model to be used', type=str)
    parse.add_argument('--name', help='The name of the run', type=str)
    parse.add_argument('--iterations', help='The number of training iterations', type=int, default=50)
    parse.add_argument('--resume', help='Whether a previous run should be resumed', action='store_true')

    args = parse.parse_args()

    # run the setup for adding custom environments and initializing ray
    setup()

    config = selected_config(args.path)

    # add the project root path to the config
    if 'env_config' not in config: config['env_config'] = {}
    if 'root' not in config['env_config']: config['env_config']['root'] = args.root or os.getcwd()

    # create an output folder for training checkpoints
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    # fetch the trainer for the requested model
    trainer = selected_trainer(args.model)

    # perform the training
    analysis = training(trainer, config, checkpoint_path, args.name, args.resume, args.iterations)

    # validate the results
    validate(analysis, trainer, config)
