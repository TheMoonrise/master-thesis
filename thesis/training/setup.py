"""
Setup class for configuring custom training parameters.
"""

import json
import numbers
import os
import yaml
import torch

import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

# tune import required for resolving hyperparameter search in config files
import ray.tune as tune  # noqa type: ignore

from ray.rllib.models.torch.torch_action_dist import TorchDirichlet
from ray.rllib.models import ModelCatalog

from thesis.training.callbacks import CustomCallbacks
from thesis.environments.grid import GridEnv
from thesis.environments.market import MarketEnv
from thesis.policies.ppo_risk_averse import risk_averse_trainer
from thesis.policies.ppo_risk_strategy import risk_averse_strategy_trainer

from crypto_markets_gym.envs.crypto_markets_env import CryptoMarketsEnv


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
        print('Beginning ray init...')
        ray.init(local_mode=debug, logging_level='error', log_to_driver=False)
        print('Ray init completed')

        # register custom environments
        register_env('Gridworld', GridEnv)
        register_env('Market', MarketEnv)
        register_env('DecentralizedMarket', CryptoMarketsEnv)

        # overwrite of the default TorchDirichlet action distribution.
        # default implementation throws and error:
        # the function kl_divergence is not implemented for torch.distributions.dirichlet
        # this patch overwrites the affected function kl with an alternative implementation
        # TODO: check if this is fixed, look for other solution
        class TorchDirichletPatch(TorchDirichlet):
            def kl(self, o):
                return torch.distributions.kl.kl_divergence(o.dist, self.dist)

        ModelCatalog.register_custom_action_dist("Dirichlet", TorchDirichletPatch)

    def shutdown(self):
        """
        Performs cleanup operations.
        """
        ray.shutdown()

    def parameters(self, path):
        """
        Reads the preferences contents from the yaml file.
        :param path: The path to the yaml config file.
        :return: The preferences dictionary.
        """
        with open(path) as file:
            # TODO: Use RlLib tuned yaml
            # run key rename -> run_or_experiment
            params = yaml.safe_load(file)

        # remove the first layer of the preferences
        params = params[list(params.keys())[0]]

        # add the default samples value
        if 'samples' not in params: params['samples'] = 1

        # modify configuration keys
        params['config']['env'] = params['env']
        params['config']['callbacks'] = CustomCallbacks

        # check if cuda is available
        if not torch.cuda.is_available():
            print('Disabled gpu usage because CUDA is not available')
            params['config']['num_gpus'] = 0

        # add the project root path to the config
        if 'env_config' in params['config'] and 'root' in params['config']['env_config']:
            root = os.path.join(os.path.dirname(__file__), '..', '..')
            params['config']['env_config']['root'] = params['config']['env_config']['root'] or root

        return params

    def update_hyperparameters(self, params, path):
        """
        Updates the parameters within the given parameters dict with the values found in another configuration.
        No string values are overwritten, only numeric values.
        This way paths, resources and settings are taken from the original preferences.
        :param params: The parameters to update
        :param path: The path to the config file to read values from.
        """
        if not os.path.exists(path): return
        with open(path) as file: config = json.load(file)

        def update_dict(dict_from, dict_to):
            for k, v in dict_from.items():
                if isinstance(v, dict):
                    if k not in dict_to: dict_to[k] = {}
                    update_dict(v, dict_to[k])

                if isinstance(v, numbers.Number) or isinstance(v, list):
                    dict_to[k] = v

        update_dict(config, params['config'])

    def hyperparameter_tuning(self, prefs):
        """
        Adds tune hyperparameter search to the preferences.
        The configuration is scanned for expressions to be evaluated.
        Any hyperparameter expression found is resolved to the actual object.
        :param prefs: The configuration dictionary to modify.
        :return: The modified prefs dictionary.
        """
        # training gets stuck if the number of trials * number of workers per trial exceeds the number of cup cores
        # it seems like after a trial terminates the cpu core is not "released"
        # when all cores have been used once the training get's stuck
        # TODO: find a solution for this
        def update_dict(dictionary):
            for k, v in dictionary.items():
                if isinstance(v, dict): update_dict(v)
                if not isinstance(v, str) or not v.startswith('tune.'): continue
                dictionary[k] = eval(v)

        update_dict(prefs['config'])
        return prefs

    def trainer(self, model):
        """
        Provides the trainer class.
        :param model: The name of the model to train.
        :return: The trainer class for the input parameters. Default return is vanilla ppo.
        """
        if model == 'PPO-RISK': return risk_averse_trainer()
        if model == 'PPO-RA-STRATEGY': return risk_averse_strategy_trainer()
        if model == 'PPO': return PPOTrainer
        raise Exception(f'Model {model} not implemented')
