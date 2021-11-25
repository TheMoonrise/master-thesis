"""
Constructs modified PPO training components to include risk management.
"""

import torch
import numpy as np

from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
# from ray.rllib.utils.torch_utils import explained_variance, sequence_mask


def risk_averse_policy():
    """
    Builds a custom policy which includes risk averse action taking.
    :return: The policy class.
    """
    policy = PPOTorchPolicy.with_updates(
        name='RiskAversePolicy',
        loss_fn=loss,
        optimizer_fn=optimizer,
        postprocess_fn=postprocessing
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


def postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Performs advantage computation. This adds risk values to each state reward.
    :param policy: The policy being trained.
    :param sample_batch: The sampled trajectories.
    :param other_agent_batches: Included for compatibility.
    :param episodes: Included for compatibility.
    :return: The updated batches dict.
    """
    with torch.no_grad():
        obs = torch.from_numpy(sample_batch[SampleBatch.OBS])

        outs_p2 = policy.model.risk_net_p2(obs).squeeze()
        outs_p1 = policy.model.risk_net_p1(obs).squeeze()

        risk = torch.maximum(torch.mean(outs_p2 - torch.pow(outs_p1, 2.0)), torch.tensor(0))
        sample_batch[SampleBatch.REWARDS] -= risk.numpy() * 1.2

        return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


def optimizer(policy, config):
    """
    Provides an optimizer for the training.
    :param policy: The policy being trained.
    :param config: The configuration dictionary of the training.
    :return: A torch optimizer.
    """
    setup_risk_nets(policy)
    return torch.optim.Adam(policy.model.parameters(), lr=config["lr"])


def setup_risk_nets(policy):
    """
    This function configures the risk nets as part of the model.
    :param policy: The policy being trained.

    """
    policy.model.risk_net_p1 = risk_net(12)
    policy.model.risk_net_p2 = risk_net(12)


def value_targets(policy, sample_batch, power=1.0):
    """
    Computes the value targets given the power to which each reward is taken.
    :param policy: The policy being trained.
    :param sample_batch: The collected trajectories.
    :param power: The power value each reward is taken to.
    """
    # get the final reward value to extend the trajectory
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    else:
        input_dict = sample_batch.get_single_step_input_dict(policy.model.view_requirements, index="last")
        last_r = policy._value(**input_dict)

    # create the combined trajectoy
    rewards_plus_v = np.concatenate([sample_batch[SampleBatch.REWARDS], np.array([last_r])])

    # raise the rewards by the given power
    rewards_plus_v = np.power(rewards_plus_v, power)

    # return the discounted reward cumsum
    # return discount_cumsum(rewards_plus_v, 0)[:-1].astype(np.float32)
    return discount_cumsum(rewards_plus_v, policy.config["gamma"])[:-1].astype(np.float32)


def loss(policy, model, dist_class, train_batch):
    """
    Custom implementation of the ppo loss function.
    This function adds risk to the objective and optimizes for variance estimation.
    :param policy: The policy that is trained.
    :param model: The model being trained.
    :param dist_class: The action distribution.
    :param train_batch: The collected training samples.
    :return: The total loss value for optimization.
    """
    trgt_p1 = torch.from_numpy(value_targets(policy, train_batch, 1))
    outs_p1 = policy.model.risk_net_p1(train_batch[SampleBatch.OBS]).squeeze()
    loss_p1 = torch.mean(torch.pow(outs_p1 - trgt_p1, 2.0))

    trgt_p2 = torch.from_numpy(value_targets(policy, train_batch, 2))
    outs_p2 = policy.model.risk_net_p2(train_batch[SampleBatch.OBS]).squeeze()
    loss_p2 = torch.mean(torch.pow(outs_p2 - trgt_p2, 2.0))

    # print(policy.model.risk_net_p1(torch.from_numpy(np.eye(12, dtype=np.float32))).squeeze().detach())

    # add the risk/variance estimators to the total loss
    loss_surroagte = ppo_surrogate_loss(policy, model, dist_class, train_batch)

    # add the loss for each risk net to the total loss
    loss = loss_surroagte + 100 * loss_p1 + 100 * loss_p2

    print('LOSS', f'{loss:10.3f}', 'LOSS SUR', f'{loss_surroagte: 10.3f}', 'LOSS P1',
          f'{loss_p1.item(): 10.3f}', 'LOSS P2', f'{loss_p2.item(): 10.3f}')

    return loss


def risk_net(input_size):
    """
    Generates a torch model for risk estimation.
    :param input_size: The size of the input layer.
    :returns: A pytorch model with the shape of input_size : 1
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 16),
        torch.nn.Linear(16, 16),
        torch.nn.Linear(16, 1)
    )
