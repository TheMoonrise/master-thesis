import gym
import os
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
# from ray.tune import trial

# create an output folder for checkpoints
# default folder is ~/ray_results
out = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(out, exist_ok=True)

# define the configuration for the agent trainer and the stopping condition
config = {'env': 'CartPole-v0', 'framework': 'torch'}
stop = {"episode_reward_mean": 160}

# run the training using tune for hyperparameter tuning
# training results are stored in analysis
analysis = tune.run(PPOTrainer, config=config, stop=stop, checkpoint_at_end=True, local_dir=out)

trial = analysis.get_best_logdir('episode_reward_mean', 'max')
ckpnt = analysis.get_best_checkpoint(trial, 'training_iteration', 'max')

# restore the agent from the checkpoint
agent = PPOTrainer(config=config)
agent.restore(ckpnt)

env = gym.make("CartPole-v0")
sta = env.reset()

while True:
    env.render()
    act = agent.compute_action(sta)
    sta, _, _, _ = env.step(act)
