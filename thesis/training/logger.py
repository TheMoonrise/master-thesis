"""
Logger implementation for tune training.
"""

from ray.tune.logger import LoggerCallback


class ProgressLogger(LoggerCallback):
    """
    A custom implementation of the LoggerCallback.
    This implementation provides minimal console output to follow the training progress.
    """

    def log_trial_result(self, iteration, trial, result):
        print(f"Trial {trial}: {iteration}")
