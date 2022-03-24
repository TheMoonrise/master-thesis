"""
Setup class for configuring custom training parameters.
"""

import os
import yaml
import torch

import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from ray.rllib.models.torch.torch_action_dist import TorchDirichlet
from ray.rllib.models import ModelCatalog

from thesis.training.callbacks import CustomCallbacks
from thesis.environments.grid import GridEnv
from thesis.policies.ppo_risk_averse import risk_averse_trainer


class Setup:
    """
    Provides utility function for configuring training and validation.
    Adds custom environments, distributions and models.
    """

    def setup(self, debug=False):
        """
        Prepares ray, and other components for training and validation.
        Register custom models with the RlLib library.
        :param debug: Whether ray should be initialized in local mode for debugging.
        """
        ray.init(local_mode=debug, logging_level='error')

        # register custom environments
        register_env('Gridworld', GridEnv)

        # overwrite of the default TorchDirichlet action distribution.
        # default implementation throws and error:
        # the function kl_divergence is not implemented for torch.distributions.dirichlet
        # this patch overwrites the affected function kl with an alternative implementation
        # TODO: check if this is fixed, look for other solution
        class TorchDirichletPatch(TorchDirichlet):
            def kl(self, o):
                return torch.distributions.kl.kl_divergence(o.dist, self.dist)

        ModelCatalog.register_custom_action_dist("Dirichlet", TorchDirichletPatch)

    def parameters(self, path):
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

        # modify configuration keys
        prefs['config']['env'] = prefs['env']
        prefs['config']['callbacks'] = CustomCallbacks

        # add the project root path to the config
        if 'env_config' in prefs['config'] and 'root' in prefs['config']['env_config']:
            prefs['config']['env_config']['root'] = prefs['config']['env_config']['root'] or os.getcwd()

        return prefs

    def trainer(self, model):
        """
        Provides the trainer class.
        :param model: The name of the model to train.
        :return: The trainer class for the input parameters. Default return is vanilla ppo.
        """
        if model == 'PPO-RISK': return risk_averse_trainer()
        if model == 'PPO': return PPOTrainer
        raise Exception(f'Model {model} not implemented')
