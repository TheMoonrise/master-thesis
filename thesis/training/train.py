"""
Training of an agent.
"""

import argparse
import os

from ray import tune
from ray.tune.utils.log import Verbosity


from thesis.training.setup import Setup
from thesis.training.logger import ProgressLogger


def training(trainer, config, stop, samples, checkpoint_path, name, resume):
    """
    Performs a training run using tune.
    :param trainer: The trainer class to be used for the training.
    :param config: The configuration for the training.
    :param stop: The stopping condition for the training.
    :param samples: The number of hyper parameter search samles to perform.
    :param checkpoint_path: The path where to save the checkpoints to.
    :param name: The name of the training run.
    :param resume: Whether a previous run with the same name should be resumed.
    :return: The analysis containing training results.
    """
    analysis = tune.run(trainer, config=config, stop=stop, checkpoint_at_end=True,
                        local_dir=checkpoint_path, resume=resume, name=name, verbose=Verbosity.V1_EXPERIMENT,
                        callbacks=[ProgressLogger()], checkpoint_freq=50, num_samples=samples)

    # return the analysis object
    return analysis


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('params', help='The configuration YAML file for the training parameters', type=str)

    parse.add_argument('--name', help='The name of the experiment', type=str)
    parse.add_argument('--resume', help='Whether a previous run should be resumed', action='store_true')
    parse.add_argument('--debug', help='When set training is single threaded', action='store_true')

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
    training(trainer, params['config'], params['stop'], params['samples'], checkpoint_path, args.name, args.resume)

    setup.shutdown()
    print('Training completed')
