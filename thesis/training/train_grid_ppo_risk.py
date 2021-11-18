"""
Training of a PPO agent on the grid world.
The agent should behave in a risk averse manner.
Risk is defined as the variance in state rewards.
"""

import argparse
import os
import time

import ray
from ray import tune

from thesis.environments.grid import GridEnv
from thesis.policies.ppo_risk_averse import risk_averse_trainer


def training(grid_path, checkpoint_path, name, resume):
    ray.init()

    # define the configuration for the training
    config = {
        "env": GridEnv,
        "env_config": {
            "grid_path": grid_path,
        },
        "num_workers": 0,
        "framework": 'torch',
    }

    stop = {
        "training_iteration": 50
    }

    # run the training using tune for hyperparameter tuning
    # training results are stored in analysis
    trainer = risk_averse_trainer()

    analysis = tune.run(trainer, config=config, stop=stop, checkpoint_at_end=True,
                        local_dir=checkpoint_path, resume=resume, name=name)

    # return the analysis object
    return analysis


def validate(analysis, grid_path):
    # perform some verification runs on the environment
    trial = analysis.get_best_logdir('episode_reward_mean', 'max')
    ckpnt = analysis.get_best_checkpoint(trial, 'training_iteration', 'max')

    # define the configuration for the validation
    config = {
        "env": GridEnv,
        "env_config": {
            "grid_path": grid_path,
        },
        "framework": 'torch',
    }

    # restore the agent from the checkpoint
    trainer = risk_averse_trainer()
    agent = trainer(config=config)
    agent.restore(ckpnt)

    env = GridEnv({'grid_path': grid_path})
    sta = env.reset()

    while True:
        act = agent.compute_action(sta)
        sta, _, done, _ = env.step(act)
        env.render()
        time.sleep(.5)

        if done:
            sta = env.reset()
            env.render()
            time.sleep(.5)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('path', help='The configuration JSON file for the grid world generation', type=str)
    parse.add_argument('--name', help='The name of the run', type=str)
    parse.add_argument('--resume', help='Whether a previous run should be resumed', action='store_true')

    args = parse.parse_args()

    # transform the relative grid path to a absolute path
    grid_path = os.path.join(os.getcwd(), args.path)

    # create an output folder for training checkpoints
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    analysis = training(grid_path, checkpoint_path, args.name, args.resume)

    # validate the results
    validate(analysis, grid_path)
