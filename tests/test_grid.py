"""
Tests for validating the functionality of the grid environment.
"""

import pytest
import json

from thesis.environments.grid import GridEnv


@pytest.fixture
def grid_setup():
    """
    Provides a default grid world setup.
    The world has the following shape.
    C A A T
    O B B T
    C C C T
    The origin is defined as O.
    The target states are defined as T.
    A, B and C define different states with negative rewards.
    :return: The setup dictionary.
    """
    setup = {
        'grid': ['AAAT', 'OBBT', 'CCCT'],
        'origin': 'O',
        'target': 'T',
        't': [+1.0, 0.0],
        'o': [-0.1, 0.0],
        'a': [-1.0, 1.0],
        'b': [-0.6, 0.0],
        'c': [-1.0, 1.0]
    }

    return json.dumps(setup)


@pytest.fixture
def grid_world(grid_setup):
    """
    Provides a grid world to test on.
    :return: The grid world.
    """
    return GridEnv({}, grid_setup)


@pytest.fixture
def grid_world_meta_actions(grid_setup):
    """
    Provides a grid world to test on that uses meta actions.
    :return: The grid world.
    """
    return GridEnv({'meta_actions': True}, grid_setup)


def test_origin_spawn(grid_world: GridEnv):
    """
    Tests whether the environment starts on the origin.
    """
    sta = grid_world.reset()
    assert sta == grid_world.state_number((1, 0))


def test_terminal_done(grid_world: GridEnv):
    """
    Tests if the state changes correctly on the terminal state.
    """
    grid_world.reset()
    grid_world.step(1)
    grid_world.step(1)

    _, _, done, _ = grid_world.step(1)
    assert done


def test_rewards(grid_world: GridEnv):
    """
    Tests if the state hands out correct rewards.
    """
    grid_world.reset()

    _, reward, _, _ = grid_world.step(1)
    assert reward < 0

    _, reward, _, _ = grid_world.step(3)
    assert reward == -0.1

    grid_world.step(1)
    grid_world.step(1)
    _, reward, _, _ = grid_world.step(1)
    assert reward == 1


def test_meta_actions(grid_world_meta_actions: GridEnv):
    """
    Tests if the meta actions work correctly.
    """
    grid_world_meta_actions.reset()
    grid_world_meta_actions.step([0.5, 0, 0.5, 0])
    grid_world_meta_actions.step([0, 1, 0, 0])
    grid_world_meta_actions.step([0, 1, 0, 0])
    grid_world_meta_actions.step([0.5, 0, 0.5, 0])

    _, _, done, _ = grid_world_meta_actions.step([0, 1, 0, 0])
    assert done
