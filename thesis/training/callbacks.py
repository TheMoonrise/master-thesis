"""
Callbacks to record additional stats during training.
"""

import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks


class CustomCallbacks(DefaultCallbacks):
    """
    Implementation of the DefaultCallbacks.
    This adds additional metrics to the already recorded data.
    Additional metrics can be viewed in Tensorboard.
    """

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies,
                                  postprocessed_batch, original_batches, **kwargs):
        """
        Injects additional logging information after each batch processing.
        :param worker: Reference to the current rollout worker.
        :param episode: Episode object.
        :param agent_id: The id of the current agent.
        :param policy_id: The id of the current policy for the agent.
        :param policies: Mapping of policy id to policy object.
        In single agent mode there will only be a single "default_policy".
        :param postprocessed_batch: The sample batch after postprocessing.
        :param original_batches: Mapping of agent to unprocessed batch data.
        """
        if 'risk' in postprocessed_batch:
            episode.custom_metrics['risk'] = np.average(postprocessed_batch['risk'])

        if 'clean_rewards' in postprocessed_batch:
            episode.custom_metrics['clean_episode_reward'] = np.average(postprocessed_batch['clean_rewards'])
