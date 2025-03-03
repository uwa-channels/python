import pytest
import numpy as np
import scipy.signal as sg
from replay_python.replay import replay


def generate_mock_channel(fs_delay=8e3, fs_time=20, fc=13e3, M=5, L=100, T=400):
    """Generate a mock channel dataset with expected structures."""
    h_hat_real = np.random.randn(T, M, L)
    h_hat_imag = np.random.randn(T, M, L)
    theta_hat = np.random.randn(np.ceil(T * fs_delay / fs_time).astype(int), M)
    params = {
        "fs_delay": np.array([[fs_delay]]),
        "fs_time": np.array([[fs_time]]),
        "fc": np.array([[fc]]),
    }
    return {
        "h_hat": {"real": h_hat_real, "imag": h_hat_imag},
        "theta_hat": theta_hat,
        "params": params,
    }


def test_replay_basic():
    fs = 96e3
    array_index = np.array([0, 1])
    channel = generate_mock_channel()
    input_signal = np.random.randn(1024)
    output = replay(input_signal, fs, array_index, channel)

    assert output.shape[1] == len(array_index), "Output channel mismatch"
    assert output.shape[0] > 0, "Output length should be greater than zero"
    assert np.isfinite(output).all(), "Output contains NaN or infinite values"


def randsamples(population, num):
    rand_index = np.random.permutation(len(population))
    return population[rand_index[:num]]


@pytest.mark.parametrize(
    "params",
    [
        {
            "fs_delay": 8e3,
            "fs_time": 20,
            "fc": 13e3,
            "Tmp": 15e-3,
            "R": 4e3,
            "channel_time": 10,
            "n_path": 10,
        },
        {
            "fs_delay": 16e3,
            "fs_time": 10,
            "fc": 10e3,
            "Tmp": 25e-3,
            "R": 4e3,
            "channel_time": 10,
            "n_path": 10,
        },
    ],
)
def test_replay_function(params):
    fs_delay = params["fs_delay"]
    fs_time = params["fs_time"]
    fc = params["fc"]
    Tmp = params["Tmp"]
    R = params["R"]
    channel_time = params["channel_time"]
    n_path = params["n_path"]

    path_delay = np.sort(randsamples(np.arange(Tmp * 1e3), n_path)) / 1e3
    path_delay -= np.min(path_delay)
    path_gain = np.exp(-path_delay / Tmp)
    c_p = path_gain * np.exp(-1j * 2 * np.pi * fc * path_delay)
    h_hat = np.zeros(
        (
            np.round(channel_time * fs_time).astype(int),
            1,
            np.ceil(fs_delay * Tmp * 1.5).astype(int),
        ),
        dtype=complex,
    )
    h_hat[:, 0, np.round((path_delay + 0.2 * Tmp) * fs_delay).astype(int)] = np.tile(c_p, (h_hat.shape[0], 1))
    h_hat = h_hat[:, :, ::-1]
    theta_hat = np.zeros((np.round(channel_time * fs_delay).astype(int), 1))

    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "theta_hat": theta_hat,
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
    }

    fs = 48e3
    data_symbols = np.random.randint(2, size=(4095,)) * 2 - 1
    baseband = sg.resample_poly(data_symbols, int(fs / R), 1)
    passband = np.real(baseband * np.exp(2j * np.pi * fc * np.arange(len(baseband)) / fs))
    input_signal = np.concatenate((np.zeros(1000), passband, np.zeros(1000)))

    r = replay(input_signal, fs, [0], channel, 1)
    v = r[:, 0] * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)

    xcor = np.abs(sg.fftconvolve(v, baseband[::-1]))
    xcor = xcor / np.max(xcor)

    peaks, _ = sg.find_peaks(
        xcor,
        height=np.min(np.abs(path_gain)) * 0.8,
        distance=np.min(np.diff(np.sort(path_delay))) * fs * 0.9,
    )

    estimated_gain = xcor[peaks]
    peaks -= np.min(peaks)
    estimated_delays = peaks / fs

    criteria = np.abs(np.sum(path_delay * path_gain) - np.sum(estimated_delays * estimated_gain))
    assert criteria < 5e-4 * n_path, f"Test criteria failed: {criteria:.3e}"


if __name__ == "__main__":
    pytest.main()
