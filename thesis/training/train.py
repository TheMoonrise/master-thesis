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

from ray.rllib.models.torch.torch_action_dist import TorchDirichlet
from ray.rllib.models import ModelCatalog

from thesis.environments.grid import GridEnv
from thesis.policies.ppo_risk_averse import risk_averse_trainer


def setup(debug):
    """
    Prepares ray, and other components for training and validation.
    Register custom models with the RlLib library.
    :param debug: Whether ray should be initialized in local mode for debugging.
    """
    ray.init(local_mode=debug)

    register_env('Gridworld', GridEnv)

    # Overwrite of the default TorchDirichlet action distribution.
    # Default implementation throws and error:
    # The function kl_divergence is not implemented for torch.distributions.dirichlet
    # This patch overwrites the affected function kl with an alternative implementation
    # TODO: Check if this is fixed, look for other solution
    class TorchDirichletPatch(TorchDirichlet):
        def kl(self, o):
            return torch.distributions.kl.kl_divergence(o.dist, self.dist)

    ModelCatalog.register_custom_action_dist("Dirichlet", TorchDirichletPatch)


def selected_preferences(path):
    """
    Reads the preferences contents from the yaml file.
    :param path: The path to the yaml config file.
    :return: The preferences dictionary.
    """
    with open(path) as file:
        # TODO: Use RlLib tuned yaml
        # run key rename -> run_or_experiment
        prefs = yaml.safe_load(file)

    # remove the first layer of the preferences
    prefs = prefs[list(prefs.keys())[0]]
    prefs['config']['env'] = prefs['env']

    # add the project root path to the config
    if 'env_config' in prefs['config'] and 'root' in prefs['config']['env_config']:
        prefs['config']['env_config']['root'] = prefs['config']['env_config']['root'] or os.getcwd()

    return prefs


def selected_trainer(model):
    """
    Provides the trainer class.
    :param model: The name of the model to train.
    :return: The trainer class for the input parameters. Default return is vanilla ppo.
    """
    if model == 'PPO-RISK': return risk_averse_trainer()
    if model == 'PPO': return PPOTrainer
    raise Exception(f'Model {model} not implemented')


def training(trainer, config, stop, checkpoint_path, name, resume):
    """
    Performs a training run using tune.
    :param trainer: The trainer class to be used for the training.
    :param config: The configuration for the training.
    :param stop: The stopping condition for the training.
    :param checkpoint_path: The path where to save the checkpoints to.
    :param name: The name of the training run.
    :param resume: Whether a previous run with the same name should be resumed.
    :return: The analysis containing training results.
    """
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
        point = analysis.get_best_checkpoint(trial, 'training_iteration', 'max')

        # restore the agent from the checkpoint
        agent = trainer(config=config)
        agent.restore(point)

        # env = GridEnv(config['env_config'])
        env_config = config['env_config'] if 'env_config' in config else {}
        sleep = env_config['timeout'] if 'timeout' in env_config else 0

        env = agent.env_creator(env_config)
        sta = env.reset()

        while True:
            act = agent.compute_single_action(sta, explore=False)
            sta, _, done, _ = env.step(act)
            env.render()
            if sleep > 0: time.sleep(sleep)

            if done:
                sta = env.reset()
                env.render()
                if sleep > 0: time.sleep(sleep)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('path', help='The configuration YAML file for the training parameters', type=str)

    parse.add_argument('--name', help='The name of the run', type=str)
    parse.add_argument('--resume', help='Whether a previous run should be resumed', action='store_true')
    parse.add_argument('--debug', help='Whether the code should not be parallelized', action='store_true')

    args = parse.parse_args()

    # run the setup for adding custom environments and initializing ray
    setup(args.debug)

    prefs = selected_preferences(args.path)

    # create an output folder for training checkpoints
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    # fetch the trainer for the requested model
    trainer = selected_trainer(prefs['run'])

    # perform the training
    analysis = training(trainer, prefs['config'], prefs['stop'], checkpoint_path, args.name, args.resume)

    # validate the results
    validate(analysis, trainer, prefs['config'])
