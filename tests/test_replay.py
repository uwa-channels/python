import pytest
import numpy as np
import scipy.signal as sg
from uwa_replay import replay
from fractions import Fraction


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


def generate_mock_channel(fs_delay=8e3, fs_time=20, fc=13e3, M=5, L=100, T=400):
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
        "version": np.array([[1.0]])
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


def test_replay_performance(benchmark):
    fs = 96e3
    array_index = np.array(np.arange(32))
    channel = generate_mock_channel(M=32, T=800)
    input_signal = np.random.randn(16384)
    result = benchmark(lambda: replay(input_signal, fs, array_index, channel))


@pytest.mark.parametrize(
    "params",
    [
        {
            "channel_time": 5,
            "coeff": 2,
            "d": 0.15,
            "fc": 10e3,
            "fs_delay": 8e3,
            "fs_time": 20,
            "has_f_resamp": False,
            "has_theta_hat": True,
            "M": 4,
            "n_path": 8,
            "R": 4e3,
            "Tmp": 10e-3,
            "velocity": 1.5,
        },
        {
            "channel_time": 5,
            "coeff": 1,
            "d": 0.15,
            "fc": 15e3,
            "fs_delay": 16e3,
            "fs_time": 20,
            "has_f_resamp": False,
            "has_theta_hat": True,
            "M": 2,
            "n_path": 10,
            "R": 8e3,
            "Tmp": 20e-3,
            "velocity": -0.5,
        },
        {
            "channel_time": 5,
            "coeff": 1.5,
            "d": 0.3,
            "fc": 15e3,
            "fs_delay": 8e3,
            "fs_time": 10,
            "has_f_resamp": False,
            "has_theta_hat": False,
            "M": 4,
            "n_path": 8,
            "R": 4e3,
            "Tmp": 20e-3,
            "velocity": 0,
        },
        {
            "channel_time": 5,
            "coeff": 1.5,
            "d": 0.2,
            "fc": 15e3,
            "fs_delay": 16e3,
            "fs_time": 10,
            "has_f_resamp": True,
            "has_theta_hat": False,
            "n_path": 8,
            "M": 4,
            "R": 8e3,
            "Tmp": 20e-3,
            "velocity": -1,
        },
        {
            "channel_time": 5,
            "coeff": 1.5,
            "d": 0.3,
            "fc": 15e3,
            "fs_delay": 10e3,
            "fs_time": 10,
            "has_f_resamp": True,
            "has_theta_hat": True,
            "M": 2,
            "n_path": 8,
            "R": 6e3,
            "Tmp": 20e-3,
            "velocity": -2,
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
    M = params["M"]

    path_delay = np.sort(randsamples(np.arange(Tmp * 1e3), n_path))[:, None] / 1e3
    incremental_delay = np.arange(M)[None, :] * params["d"] / 1545
    path_delay = path_delay + incremental_delay
    path_delay -= np.min(path_delay)
    path_gain = np.exp(-path_delay * params["coeff"] / Tmp)
    c_p = path_gain * np.exp(-1j * 2 * np.pi * fc * path_delay)
    h_hat = np.zeros(
        (
            M,
            np.ceil(fs_delay * Tmp * 1.5).astype(int),
        ),
        dtype=complex,
    )
    rows = np.tile(np.arange(M), n_path)
    cols = np.round((path_delay + 0.2 * Tmp) * fs_delay).astype(int).ravel()
    h_hat[rows, cols] = c_p.ravel()
    h_hat = np.tile(h_hat[None, :, :], (np.round(channel_time * fs_time).astype(int), 1, 1))

    f_resamp = 1 / (1 + params["velocity"] / 1545)
    a = 1 - 1 / f_resamp
    t = np.arange(np.round(channel_time * fs_delay).astype(int))
    theta_hat = -a * 2 * np.pi * fc * t[:, None] / fs_delay
    theta_hat = np.tile(theta_hat, (1, M))

    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
        "version": np.array([[1.0]]),
    }

    if params["has_theta_hat"]:
        channel["theta_hat"] = theta_hat
    if params["has_f_resamp"]:
        channel["f_resamp"] = np.array([[f_resamp]])
    if params["has_theta_hat"] and params["has_f_resamp"]:
        channel["theta_hat"] = np.zeros(theta_hat.shape)
        channel["f_resamp"] = np.array([[f_resamp]])

    fs = 48e3
    data_symbols = np.random.randint(2, size=(4095,)) * 2 - 1
    baseband = sg.resample_poly(data_symbols, int(fs / R), 1)
    passband = np.real(baseband * np.exp(2j * np.pi * fc * np.arange(len(baseband)) / fs))
    input_signal = np.concatenate((np.zeros(1000), passband, np.zeros(1000)))

    r = replay(input_signal, fs, np.arange(M), channel, 1)
    sync = np.zeros((M,))
    for m in range(M):
        v = r[:, m] * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)

        frac = Fraction(f_resamp).limit_denominator()
        baseband_resampled = baseband * np.exp(-2j * a * np.pi * fc * np.arange(len(baseband)) / fs)
        baseband_resampled = sg.resample_poly(baseband_resampled, frac.numerator, frac.denominator)

        xcor = np.abs(sg.fftconvolve(v, baseband_resampled[::-1].conj()))
        xcor = xcor / np.max(xcor)

        peaks, _ = sg.find_peaks(
            xcor,
            height=np.min(np.abs(path_gain[:, m])) * 0.8,
            distance=np.min(np.diff(np.sort(path_delay[:, m]))) * fs * 0.9,
        )
        sync[m] = peaks[0]

        estimated_gain = xcor[peaks]
        peaks -= np.min(peaks)
        estimated_delays = peaks / fs

        # import matplotlib.pyplot as plt
        # lags = np.arange(len(xcor)) - np.argmax(xcor)
        # plt.stem(path_delay[:, m] * 1e3, path_gain[:, m], markerfmt="x", basefmt=" ")
        # plt.plot((lags+sync[m]-sync[0]) / fs * 1e3, xcor)
        # plt.xlim([np.min(estimated_delays) * 1e3 - 5, np.max(estimated_delays) * 1e3 + 10])
        # plt.xlabel("Delay [ms]")
        criteria = np.abs(np.sum(path_delay[:, m] * path_gain[:, m]) - np.sum(estimated_delays * estimated_gain))
        assert criteria < 3e-4 * n_path, f"Test criteria failed: {criteria:.3e}"
    # plt.show()


if __name__ == "__main__":
    pytest.main()
