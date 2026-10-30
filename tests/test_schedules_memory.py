import numpy as np
import pytest

from blobquest.config import ScheduleConfig
from blobquest.memory import ReplayMemory, Transition
from blobquest.schedules import ConstantSchedule, ExponentialSchedule, LinearSchedule, make_schedule


def test_linear_schedule() -> None:
    s = make_schedule(ScheduleConfig(kind="linear", start=1.0, end=0.0, decay_fraction=0.5), total_episodes=100)
    assert isinstance(s, LinearSchedule)
    assert s(0) == pytest.approx(1.0)
    assert s(25) == pytest.approx(0.5)
    assert s(50) == pytest.approx(0.0)
    assert s(99) == pytest.approx(0.0)


def test_exponential_schedule() -> None:
    s = make_schedule(ScheduleConfig(kind="exponential", start=1.0, end=0.1, decay_rate=0.5), total_episodes=10)
    assert isinstance(s, ExponentialSchedule)
    assert s(0) == pytest.approx(1.0)
    assert s(1) == pytest.approx(0.1 + 0.9 * 0.5)
    assert s(1000) == pytest.approx(0.1)


def test_constant_schedule() -> None:
    s = make_schedule(ScheduleConfig(kind="constant", start=0.2, end=0.2), total_episodes=10)
    assert isinstance(s, ConstantSchedule)
    assert s(0) == s(9) == 0.2


def _t(reward: float, state: int = 0) -> Transition:
    return Transition(state, 0, reward, 0, 0, False)


def test_memory_ring_buffer_overwrites_oldest() -> None:
    mem = ReplayMemory(capacity=3, mistake_threshold=-10.0, mistake_weight=4.0, rng=np.random.default_rng(0))
    for i in range(5):
        mem.add(_t(0.0, state=i))
    assert len(mem) == 3
    assert {t.state for t in mem.sample(200)} == {2, 3, 4}


def test_memory_prefers_mistakes() -> None:
    mem = ReplayMemory(capacity=10, mistake_threshold=-10.0, mistake_weight=9.0, rng=np.random.default_rng(0))
    mem.add(_t(-100.0, state=1))
    mem.add(_t(-1.0, state=2))
    assert mem.num_mistakes == 1
    picks = [t.state for t in mem.sample(5000)]
    assert 0.85 < picks.count(1) / len(picks) < 0.95


def test_empty_memory_samples_nothing() -> None:
    mem = ReplayMemory(capacity=3, mistake_threshold=-10.0, mistake_weight=4.0, rng=np.random.default_rng(0))
    assert mem.sample(10) == []
