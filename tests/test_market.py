"""
Tests for validating the functionality of the market environment.
"""

import math
import pytest
import numpy as np

from thesis.environments.market import MarketEnv


@pytest.fixture
def market_setup():
    """
    Provides a market data string.
    :return: The data string.
    """
    setup = ('timestamp,AAA,BBB,CCCC, DDD\n'
             '1600000200000,1.5,100,3856.2388934,0\n'
             '1600000500000,3.0,200,4078.2903476,1\n'
             '1600000800000,6.0,100,3998.9234796,2\n'
             '1600001100000,2.0,200,3823.2346034,3\n'
             '1600001400000,1.0,100,3754.8845328,4\n'
             '1600001700000,4.0,200,3899.2345834,5\n'
             '1600002000000,1.5,100,4003.1290874,6\n')

    return setup


def test_environment_steps(market_setup):
    """
    Tests whether the environment steps correctly.
    :param market_setup: The setup string for the environment.
    """
    fee = 0.1
    config = {'episode_length': 3, 'validation_split': 1, 'is_validation': True, 'transaction_fee': fee}
    market_env = MarketEnv(config, market_setup)

    assert market_env.done

    state = market_env.reset()
    assert all(state == np.array([1, 1.5, 100, 3856.2388934, 0, 0]))

    state, reward, done, _ = market_env.step(0)
    assert all(state == np.array([1, 3.0, 200, 4078.2903476, 1, 0]))
    assert reward == 0
    assert not done

    state, reward, done, _ = market_env.step(1)
    assert all(state == np.array([1, 6.0, 100, 3998.9234796, 2, 1]))
    assert reward == math.log(6 / 3) + math.log(1 - fee)
    assert done

    state = market_env.reset()
    assert all(state == np.array([1, 2.0, 200, 3823.2346034, 3, 0]))


def test_environment_setup(market_setup):
    """
    Tests whether the environment performs the setup correctly.
    :param market_setup: The setup string for the environment.
    """
    config = {'episode_length': 7, 'validation_split': 0}
    market_env = MarketEnv(config, market_setup)
    assert market_env.data.shape == (1, 7, 5)

    config = {'episode_length': 3, 'validation_split': 0.5}
    market_env = MarketEnv(config, market_setup)
    assert market_env.data.shape == (1, 3, 5)

    config = {'episode_length': 2, 'validation_split': 0.4}
    market_env = MarketEnv(config, market_setup)
    assert market_env.data.shape == (2, 2, 5)
    assert all(market_env.data[1, 0] == np.array([1, 6, 100, 3998.9234796, 2]))

    config = {'episode_length': 2, 'validation_split': 0.8, 'is_validation': True}
    market_env = MarketEnv(config, market_setup)
    assert market_env.data.shape == (2, 2, 5)
    assert all(market_env.data[0, 0] == np.array([1, 6, 100, 3998.9234796, 2]))
    assert all(market_env.order == np.arange(2))
