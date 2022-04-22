"""
Training of an agent.
"""

import argparse
import os

from ray import tune
from ray.tune.utils.log import Verbosity
from ray.tune.integration.wandb import WandbLoggerCallback

from thesis.training.setup import Setup
from thesis.training.logger import ProgressLogger


def training(trainer, config, stop, samples, checkpoint_path, name, restore, offline):
    """
    Performs a training run using tune.
    :param trainer: The trainer class to be used for the training.
    :param config: The configuration for the training.
    :param stop: The stopping condition for the training.
    :param samples: The number of hyper parameter search samles to perform.
    :param checkpoint_path: The path where to save the checkpoints to.
    :param name: The name of the training run.
    :param restore: Path to the checkpoint that should be restored.
    :param offline: Whether wandb logging should be disabled.
    :return: The analysis containing training results.
    """
    callbacks = [ProgressLogger()]

    if not offline:
        # configure logging to weights & biases
        # for wandb to work the file ./ray/tune/integration/wandb.py must be modified
        # in os.environ["WANDB_START_METHOD"] = "fork" the fork must be replaced by thread for windows
        wandb_key = os.path.join(os.path.dirname(__file__), '..', '..', 'wandb.key')

        # the wandb extension seems to cause bluescreens on my machine
        # according to the memory dump this is related to illegal memory writes of the graphics card driver
        # this happen during regular training and when aborting runs
        # because of this wandb is not really useful at this point in time
        # TODO: investigate bluescreen page fault in nonpaged area nvlddmkm.sys caused by python.exe
        callbacks.append(WandbLoggerCallback('MasterThesis', api_key_file=wandb_key, log_config=False, group=name))

    analysis = tune.run(trainer, config=config, stop=stop, checkpoint_at_end=True,
                        local_dir=checkpoint_path, name=name, verbose=Verbosity.V1_EXPERIMENT,
                        callbacks=callbacks, checkpoint_freq=50, num_samples=samples, restore=restore)

    # return the analysis object
    return analysis


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('params', help='The configuration YAML file for the training parameters', type=str)

    parse.add_argument('--name', help='The name of the experiment', type=str)
    parse.add_argument('--restore', help='The path to the checkpoint to restore before training', type=str)
    parse.add_argument('--debug', help='When set training is single threaded', action='store_true')
    parse.add_argument('--wandb', help='When set wandb logging is enabled', action='store_true')

    args = parse.parse_args()

    print('Beginning training setup...')

    # run the setup for adding custom environments and initializing ray
    setup = Setup()
    setup.setup(args.debug)

    params = setup.parameters(args.params)
    params = setup.hyperparameter_tuning(params)

    # create an output folder for training checkpoints
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    # fetch the trainer for the requested model
    trainer = setup.trainer(params['run'])

    # perform the training
    name = args.name or params['name'] if 'name' in params else None
    offline = not args.wandb or args.debug

    training(trainer, params['config'], params['stop'], params['samples'], checkpoint_path, name, args.restore, offline)

    setup.shutdown()
    print('Training completed')
