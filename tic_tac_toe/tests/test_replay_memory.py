from collections import deque

import pytest

from tic_tac_toe.agent.memory.replay import StandardReplayMemory

@pytest.fixture
def std_rm():
    return StandardReplayMemory(capacity=3)

def test_invalid_maxlen():
    with pytest.raises(ValueError):
        StandardReplayMemory(capacity=-1)

def test_memory_construction(std_rm):
    assert deque(maxlen=3) == std_rm.memory

def test_add_one_element(std_rm):
    std_rm.push(1)
    assert 1 == std_rm.memory[0]

def test_add_max_num_elements(std_rm):
    std_rm.push(1)
    std_rm.push(3)
    std_rm.push(2)
    assert [1, 3, 2] == list(std_rm.memory)

def test_add_element_when_full_shifts_all(std_rm):
    std_rm.push(1)
    std_rm.push(2)
    std_rm.push(3)
    std_rm.push(4)
    assert [2, 3, 4] == list(std_rm.memory)

def test_sample_empty(std_rm):
    with pytest.raises(ValueError):
        std_rm.sample(1)

def test_sample_non_empty(std_rm):
    std_rm.push('a')
    assert ['a'] == std_rm.sample(1)

def test_sample_more_than_exists(std_rm):
    std_rm.push('a')
    with pytest.raises(ValueError):
        std_rm.sample(2)

def test_can_provide_positive(std_rm):
    std_rm.push(1)
    std_rm.push(2)
    std_rm.push(3)
    assert std_rm.can_provide(3) == True

def test_can_provide_negative(std_rm):
    std_rm.push(1)
    std_rm.push(2)
    std_rm.push(3)
    assert std_rm.can_provide(4) == False
