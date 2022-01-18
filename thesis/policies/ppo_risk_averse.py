"""
Constructs modified PPO training components to include risk management.
"""

import torch
import numpy as np

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss
from ray.rllib.policy.sample_batch import SampleBatch
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
        postprocess_fn=postprocessing_fn
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
        obs = torch.from_numpy(sample_batch[SampleBatch.OBS])

        outs_p2 = policy.model.risk_net_p2(obs).squeeze()
        outs_p1 = policy.model.risk_net_p1(obs).squeeze()

        # risk = torch.maximum(torch.mean(outs_p2 - torch.pow(outs_p1, 2.0)), torch.tensor(0))
        risk = torch.maximum(outs_p2 - torch.pow(outs_p1, 2.0), torch.tensor(0))

        sample_batch['CLEANREWARDS'] = sample_batch[SampleBatch.REWARDS].copy()
        sample_batch[SampleBatch.REWARDS] -= risk.numpy() * 1.2

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
    input_size = policy.observation_space.shape[0]

    policy.model.risk_net_p1 = risk_net(input_size, config)
    policy.model.risk_net_p2 = risk_net(input_size, config)

    # define optimizers for each risk estimation model
    optim_risk_p1 = torch.optim.Adam(policy.model.risk_net_p1.parameters(), lr=config["lr"])
    optim_risk_p2 = torch.optim.Adam(policy.model.risk_net_p2.parameters(), lr=config["lr"])

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
    trgt_p1 = train_batch['CLEANREWARDS']
    outs_p1 = policy.model.risk_net_p1(train_batch[SampleBatch.OBS]).squeeze()
    loss_p1 = torch.mean(torch.pow(outs_p1 - trgt_p1, 2.0))

    trgt_p2 = torch.pow(train_batch['CLEANREWARDS'], 2.0)
    outs_p2 = policy.model.risk_net_p2(train_batch[SampleBatch.OBS]).squeeze()
    loss_p2 = torch.mean(torch.pow(outs_p2 - trgt_p2, 2.0))

    input_size = policy.observation_space.shape[0]
    b = policy.model.risk_net_p2(torch.from_numpy(np.eye(input_size, dtype=np.float32))).squeeze().detach().numpy()
    a = policy.model.risk_net_p1(torch.from_numpy(np.eye(input_size, dtype=np.float32))).squeeze().detach().numpy()

    print('RISK BY STATE (VARIANCE ESTIMATE)\n', np.round(b - np.power(a, 2), 2))
    print('MEAN BY STATE\n', np.round(a, 2))

    # obs = np.argmax(train_batch[SampleBatch.OBS].numpy(), axis=1)
    # next_obs = np.argmax(train_batch[SampleBatch.NEXT_OBS].numpy(), axis=1)
    # actions = np.array([str(x) for x in train_batch[SampleBatch.ACTIONS].numpy()])

    # print('DATA\n', np.stack((obs, next_obs, actions), axis=1))

    # compute the default loss value
    loss_surrogate = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    # print('LOSS SUR', f'{loss_surrogate: 10.3f}', 'LOSS P1', f'{loss_p1.item(): 10.3f}', 'LOSS P2', f'{loss_p2.item(): 10.3f}')

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
