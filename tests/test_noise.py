import pytest
import numpy as np
from uwa_replay import noisegen


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


def generate_mock_noise(fs, M=4, sigma=None):
    if sigma is None:
        sigma = np.eye(M)
    return {
        "Fs": np.array([[fs]]),
        "sigma": sigma,
        "h": np.random.randn(M, 100),
        "version": 1.0,
    }


@pytest.mark.parametrize("noise_option", [1, 2])
@pytest.mark.parametrize("M", np.arange(4))
def test_noisegen_valid_options(noise_option, M):
    input_signal = np.zeros((1000, M))
    fs = 48000

    if noise_option == 1:
        w = noisegen(input_signal.shape, fs)
    else:
        array_index = np.arange(M)
        noise = generate_mock_noise(fs, M)
        w = noisegen(input_signal.shape, fs, array_index, noise)

    assert w.shape == input_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(w)), "Output contains NaN or infinite values"


def generate_mock_impulsive_noise(fs, alpha=1.7):
    return {
        "Fs": np.array([[fs]]),
        "alpha": alpha,
        "beta": np.repeat(np.diag([1, 2, 3])[None, :, :], 65, axis=0),
    }


def test_generate_impulsive_noise_valid_options():
    input_signal = np.zeros((1000, 3))
    fs = 48000
    array_index = np.arange(3)
    noise = generate_mock_impulsive_noise(fs)
    w = noisegen(input_signal.shape, fs, array_index, noise)

    assert w.shape == input_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(w)), "Output contains NaN or infinite values"
