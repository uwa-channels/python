import pytest
import numpy as np
from uwa_replay import generate_noise


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
def test_generate_noise_valid_options(noise_option, M):
    input_signal = np.zeros((1000, M))
    fs = 48000

    if noise_option == 1:
        w = generate_noise(input_signal.shape, fs)
    else:
        array_index = np.arange(M)
        noise = generate_mock_noise(fs, M)
        w = generate_noise(input_signal.shape, fs, noise, array_index)

    assert w.shape == input_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(w)), "Output contains NaN or infinite values"


@pytest.mark.parametrize("M", np.arange(1, 4))
def test_power_normalization_option1(M):
    input_signal = np.zeros((int(1e6), M))
    fs = 90000

    w = generate_noise(input_signal.shape, fs)
    p = np.mean(np.abs(w) ** 2, axis=0)

    assert np.all(np.abs(p - p[0]) < 1e-3), "Power normalization failed"


@pytest.mark.parametrize("M", np.arange(1, 4))
def test_power_normalization_option2(M):
    input_signal = np.zeros((int(1e6), M))
    array_index = np.arange(M)
    fs = 20000
    noise = generate_mock_noise(fs, M, sigma=np.zeros((M, M)))
    np.fill_diagonal(noise["sigma"], np.random.rand(M))

    w = generate_noise(input_signal.shape, fs, noise, array_index)
    p = np.mean(np.abs(w) ** 2, axis=0)

    diag = np.copy(np.diag(noise["sigma"]))
    diag /= np.sum(diag)
    p /= np.sum(p)
    metric = p / np.diag(noise["sigma"])

    assert np.all(np.abs(metric - metric[0]) < 3e-2), "Power normalization check failed"

def generate_mock_impulsive_noise(fs, alpha=1.7):
    return {
        "Fs": np.array([[fs]]),
        "alpha": alpha,
        "beta": np.repeat(np.diag([1,2,3])[None, :, :], 65, axis=0)
    }


def test_generate_impulsive_noise_valid_options():
    input_signal = np.zeros((1000, 3))
    fs = 48000
    array_index = np.arange(3)
    noise = generate_mock_impulsive_noise(fs)
    w = generate_noise(input_signal.shape, fs, noise, array_index)

    assert w.shape == input_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(w)), "Output contains NaN or infinite values"
