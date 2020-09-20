import pytest
import numpy as np

from tic_tac_toe.agent.strategy.epsilon_greedy import EpsilonGreedyStrategy

def test_error_with_empty_arg():
    s = EpsilonGreedyStrategy(start=0, end=0, decay=0)
    with pytest.raises(ValueError):
        s.select_action(np.array([]))

def test_error_when_end_larger_than_start():
    with pytest.raises(ValueError):
        EpsilonGreedyStrategy(start=0, end=1, decay=0)

def test_error_when_negative_decay():
    with pytest.raises(ValueError):
        EpsilonGreedyStrategy(start=1, end=0, decay=-0.1)

def test_error_when_negative_end():
    with pytest.raises(ValueError):
        EpsilonGreedyStrategy(start=1, end=-1, decay=0.1)

def test_zero_exploration_returns_max():
    s = EpsilonGreedyStrategy(start=0, end=0, decay=0)
    aq_pairs = np.array([(0, -1), (1, 5), (3, 4)])
    assert 1 == s.select_action(aq_pairs)

def test_returns_choice_of_possible_actions():
    s = EpsilonGreedyStrategy(start=1, end=1, decay=0)
    aq_pairs = np.array([(0, 5), (1, 5), (3, 4)])
    assert s.select_action(aq_pairs) in (0, 1, 3)

def test_decay_decreases_exploration_rate():
    s = EpsilonGreedyStrategy(start=1, end=0, decay=.5)
    aq_pairs = np.array([(0, 3)])
    r1 = s.get_decayed_rate()
    s.select_action(aq_pairs)
    assert r1 > s.get_decayed_rate()
