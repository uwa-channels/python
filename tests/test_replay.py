import pytest
import numpy as np
from replay_python.replay import replay, pwr


def generate_mock_channel(fs_delay=8e3, fs_time=20, fc=13e3, M=5, L=100, T=400):
    """Generate a mock channel dataset with expected structures."""
    h_hat_real = np.random.randn(T, M, L)
    h_hat_imag = np.random.randn(T, M, L)
    theta_hat = np.random.randn(np.ceil(T * fs_delay / fs_time).astype(int), M)
    params = {"fs_delay": np.array([[fs_delay]]), "fs_time": np.array([[fs_time]]), "fc": np.array([[fc]])}
    return {"h_hat": {"real": h_hat_real, "imag": h_hat_imag}, "theta_hat": theta_hat, "params": params}


def test_replay_basic():
    fs = 96e3
    array_index = np.array([0, 1])
    channel = generate_mock_channel()
    input_signal = np.random.randn(1024)
    output = replay(input_signal, fs, array_index, channel)

    assert output.shape[1] == len(array_index), "Output channel mismatch"
    assert output.shape[0] > 0, "Output length should be greater than zero"
    assert np.isfinite(output).all(), "Output contains NaN or infinite values"


if __name__ == "__main__":
    pytest.main()
