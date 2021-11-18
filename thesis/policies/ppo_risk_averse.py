"""
Constructs modified PPO training components to include risk management.
"""

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy


def risk_averse_policy():
    """
    Builds a custom policy which includes risk averse action taking.
    :return: The policy object.
    """
    policy = PPOTorchPolicy.with_updates(
        name='RiskAversePolicy'
    )

    return policy


def risk_averse_trainer():
    """
    Wraps the risk averse policy into a trainer.
    :return: The trainer object for the custom risk averse policy.
    """
    policy = risk_averse_policy()

    trainer = PPOTrainer.with_updates(
        default_policy=policy
    )

    return trainer
