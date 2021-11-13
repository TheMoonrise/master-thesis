"""
Training of a PPO agent on the grid world.
"""

import argparse
import os
import time

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from thesis.environments.grid import GridEnv


if __name__ == '__main__':
    # ray.init()

    parse = argparse.ArgumentParser()
    parse.add_argument('path', help='The configuration JSON file for the grid world generation')
    args = parse.parse_args()

    # create an output folder for training checkpoints
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    # transform the relative grid path to a absolute path
    grid_path = os.path.join(os.getcwd(), args.path)

    # define the configuration for the training
    config = {
        "env": GridEnv,
        "env_config": {
            "grid_path": grid_path,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "num_workers": 1,
        "framework": 'torch',
    }

    stop = {
        "training_iteration": 10
    }

    # run the training using tune for hyperparameter tuning
    # training results are stored in analysis
    analysis = tune.run(PPOTrainer, config=config, stop=stop, checkpoint_at_end=True, local_dir=checkpoint_path)

    # perform some verification runs on the environment
    trial = analysis.get_best_logdir('episode_reward_mean', 'max')
    ckpnt = analysis.get_best_checkpoint(trial, 'training_iteration', 'max')

    # restore the agent from the checkpoint
    agent = PPOTrainer(config=config)
    agent.restore(ckpnt)

    env = GridEnv(grid_path)
    sta = env.reset()

    while True:
        act = agent.compute_action(sta)
        sta, _, _, _ = env.step(act)
        env.render()
        time.sleep(1)
