"""
Constructs modified PPO training components to include risk management.
"""

import torch
import numpy as np

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss, kl_and_loss_stats, setup_mixins
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
# from ray.rllib.models.torch.torch_action_dist import TorchDirichlets
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
# from ray.rllib.utils.torch_utils import explained_variance, sequence_mask


def risk_averse_policy():
    """
    Builds a custom policy which includes risk averse action taking.
    :return: The policy class.
    """
    policy = PPOTorchPolicy.with_updates(
        name='RiskAversePolicy',
        loss_fn=loss_fn,
        optimizer_fn=optimizer_fn,
        postprocess_fn=postprocessing_fn,
        stats_fn=stats_fn,
        before_loss_init=after_init
    )

    return policy


def risk_averse_trainer():
    """
    Wraps the risk averse policy into a trainer.
    :return: The trainer class for the custom risk averse policy.
    """
    policy = risk_averse_policy()

    trainer = PPOTrainer.with_updates(
        default_policy=policy,
        get_policy_class=lambda _: None
    )

    return trainer


def stats_fn(policy, train_batch):
    """
    Adds additional stats to the tracking.
    This is called after the learn_on_batch functions.
    :param policy: The policy that is being trained.
    :param train_batch: The final train batch.
    :return: A dictionary containing the policy stats.
    """
    stats = kl_and_loss_stats(policy, train_batch)
    stats['moment_loss_1'] = train_batch['moment_loss_1'].item()
    stats['moment_loss_2'] = train_batch['moment_loss_2'].item()

    return stats


def after_init(policy, obs_space, action_space, config):
    """
    Called after the policy is initialized.
    Additional view requirements are added in this step.
    :param policy: The active policy.
    :param obs_space: The observation space of the policy.
    :param action_space: The action space of the policy.
    :config: The trainer configuration dict.
    """
    policy.view_requirements['clean_rewards'] = ViewRequirement(data_col='rewards', shift=0)
    setup_mixins(policy, obs_space, action_space, config)


def postprocessing_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Performs advantage computation. This adds risk values to each state reward.
    :param policy: The policy being trained.
    :param sample_batch: The sampled trajectories.
    :param other_agent_batches: Included for compatibility.
    :param episodes: Included for compatibility.
    :return: The updated batches dict.
    """
    with torch.no_grad():
        actions = sample_batch[SampleBatch.ACTIONS]
        if len(actions.shape) < 2: actions = np.reshape(actions, (-1, 1))

        risk_in = torch.from_numpy(np.concatenate((sample_batch[SampleBatch.OBS], actions), axis=1)).float()
        sample_batch['risk_input'] = risk_in.to(policy.device)

        outs_p2 = policy.model.risk_net_p2(sample_batch['risk_input']).squeeze().to('cpu')
        outs_p1 = policy.model.risk_net_p1(sample_batch['risk_input']).squeeze().to('cpu')

        # risk = torch.maximum(torch.mean(outs_p2 - torch.pow(outs_p1, 2.0)), torch.tensor(0))
        risk = torch.maximum(outs_p2 - torch.pow(outs_p1, 2.0), torch.tensor(0))
        risk_factor = policy.config['model']['custom_model_config']['risk_factor']

        # check if risk is single value; if yes unsqueeze to match shape of rewards
        sample_batch['risk'] = risk.numpy() if risk.shape else torch.unsqueeze(risk, 0).numpy()
        sample_batch[SampleBatch.REWARDS] -= sample_batch['risk'] * risk_factor

        return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


def optimizer_fn(policy, config):
    """
    Provides an optimizer for the training.
    :param policy: The policy being trained.
    :param config: The configuration dictionary of the training.
    :return: A torch optimizer.
    """

    # generate the default optimizer for the policy net
    optim = torch.optim.Adam(policy.model.parameters(), lr=config["lr"])

    # attache the risk estimation models to the policy model
    # these models must be part of the policy model for parameter assignment to work
    action_size = policy.action_space.shape[0] if policy.action_space.shape else 1
    observation_size = policy.observation_space.shape[0]
    input_size = observation_size + action_size

    policy.model.risk_net_p1 = risk_net(input_size, config).to(policy.device)
    policy.model.risk_net_p2 = risk_net(input_size, config).to(policy.device)

    # define optimizers for each risk estimation model
    risk_lr = config['model']['custom_model_config']['risk_lr']
    optim_risk_p1 = torch.optim.Adam(policy.model.risk_net_p1.parameters(), lr=risk_lr)
    optim_risk_p2 = torch.optim.Adam(policy.model.risk_net_p2.parameters(), lr=risk_lr)

    return optim, optim_risk_p1, optim_risk_p2


def loss_fn(policy, model, dist_class, train_batch):
    """
    Custom implementation of the ppo loss function.
    :param policy: The policy that is trained.
    :param model: The model being trained.
    :param dist_class: The action distribution.
    :param train_batch: The collected training samples.
    :return: The total loss value for optimization.
    """
    # compute the loss for each of the risk estimation networks
    trgt_p1 = train_batch['clean_rewards']
    trgt_p2 = torch.pow(trgt_p1, 2.0)

    outs_p1 = policy.model.risk_net_p1(train_batch['risk_input']).squeeze()
    loss_p1 = torch.mean(torch.pow(outs_p1 - trgt_p1, 2.0))

    outs_p2 = policy.model.risk_net_p2(train_batch['risk_input']).squeeze()
    loss_p2 = torch.mean(torch.pow(outs_p2 - trgt_p2, 2.0))

    train_batch['moment_loss_1'] = torch.unsqueeze(loss_p1.detach(), 0)
    train_batch['moment_loss_2'] = torch.unsqueeze(loss_p2.detach(), 0)

    # compute the default loss value
    loss_surrogate = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    # print('LOSS SUR', f'{loss_surrogate: 10.3f}', 'LP1', f'{loss_p1.item(): 10.3f}', 'LP2', f'{loss_p2.item(): 10.3f}')

    return loss_surrogate, loss_p1, loss_p2


def risk_net(input_size, config):
    """
    Generates a torch model for risk estimation.
    :param input_size: The size of the input layer.
    :param config: The training configuration.
    :returns: A pytorch model with the shape of input_size : 1
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1)
    )
