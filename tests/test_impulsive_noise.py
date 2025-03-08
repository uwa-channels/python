import pytest
import numpy as np
from uwa_replay import generate_impulsive_noise


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


def generate_mock_noise(fs, alpha=1.7):
    return {
        "Fs": np.array([[fs]]),
        "alpha": alpha,
        "beta": np.repeat(np.diag([1,2,3])[None, :, :], 65, axis=0)
    }


def test_generate_impulsive_noise_valid_options():
    input_signal = np.zeros((1000, 3))
    fs = 48000
    array_index = np.arange(3)
    noise = generate_mock_noise(fs)
    w = generate_impulsive_noise(input_signal.shape, fs, noise, array_index)

    assert w.shape == input_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(w)), "Output contains NaN or infinite values"
