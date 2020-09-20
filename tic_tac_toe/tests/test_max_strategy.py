import pytest
import numpy as np

from tic_tac_toe.agent.strategy.max_strategy import MaxStrategy

@pytest.fixture
def max_stg():
    return MaxStrategy()

def test_returns_max(max_stg):
    aq_pairs = np.array([(0, -1), (1, 5), (3, 4)])
    action = max_stg.select_action(aq_pairs)
    assert 1 == action

def test_returns_first_when_tie(max_stg):
    aq_pairs = np.array([(0, 5), (1, 5), (3, 4)])
    action = max_stg.select_action(aq_pairs)
    assert 0 == action

def test_error_with_empty_arg(max_stg):
    with pytest.raises(ValueError):
        max_stg.select_action(np.array([]))
