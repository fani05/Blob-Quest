import pytest

from blobquest.state import State, StateEncoder, energy_bucket


def test_encode_decode_roundtrip_covers_every_index() -> None:
    enc = StateEncoder(rows=3, cols=4, num_buckets=3, num_boosters=2)
    assert enc.num_states == 3 * 4 * 3 * 4 * 2 * 2
    seen = set()
    for idx in range(enc.num_states):
        state = enc.decode(idx)
        assert enc.encode(state) == idx
        seen.add(state)
    assert len(seen) == enc.num_states


def test_encoding_matches_numpy_layout() -> None:
    import numpy as np

    enc = StateEncoder(rows=5, cols=6, num_buckets=4, num_boosters=3)
    state = State(2, 5, 3, 6, True, False)
    expected = np.ravel_multi_index((2, 5, 3, 6, 1, 0), enc.shape)
    assert enc.encode(state) == expected


def test_decode_out_of_range() -> None:
    enc = StateEncoder(2, 2, 1, 0)
    with pytest.raises(ValueError):
        enc.decode(enc.num_states)


@pytest.mark.parametrize(
    ("energy", "expected"),
    [(-5, 0), (0, 0), (11, 0), (12, 0), (13, 1), (30, 2), (59, 4), (60, 4), (99, 4)],
)
def test_energy_bucket(energy: int, expected: int) -> None:
    assert energy_bucket(energy, maximum=60, num_buckets=5) == expected
