import pytest
import numpy as np
from uwa_replay import  unpack


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


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
            "channel_time": 5,
            "n_path": 10,
            "has_theta_hat": False,
            "has_f_resamp": False,
            "f_resamp": 1 / (1 + 1 / 1545),
        },
        {
            "fs_delay": 16e3,
            "fs_time": 10,
            "fc": 10e3,
            "Tmp": 25e-3,
            "R": 4e3,
            "channel_time": 10,
            "n_path": 10,
            "has_theta_hat": True,
            "has_f_resamp": False,
            "f_resamp": 1 / (1 + 1.5 / 1545),
        },
        {
            "fs_delay": 16e3,
            "fs_time": 10,
            "fc": 10e3,
            "Tmp": 25e-3,
            "R": 4e3,
            "channel_time": 10,
            "n_path": 10,
            "has_theta_hat": True,
            "has_f_resamp": False,
            "f_resamp": 1 / (1 - 1 / 1545),
        },
        {
            "fs_delay": 16e3,
            "fs_time": 10,
            "fc": 10e3,
            "Tmp": 25e-3,
            "R": 4e3,
            "channel_time": 10,
            "n_path": 10,
            "has_theta_hat": False,
            "has_f_resamp": True,
            "f_resamp": 1 / (1 - 1 / 1545),
        },
        {
            "fs_delay": 16e3,
            "fs_time": 10,
            "fc": 10e3,
            "Tmp": 25e-3,
            "R": 4e3,
            "channel_time": 10,
            "n_path": 10,
            "has_theta_hat": True,
            "has_f_resamp": True,
            "f_resamp": 1 / (1 - 1 / 1545),
        },
    ],
)
def test_unpack_function(params):
    fs_delay = params["fs_delay"]
    fs_time = params["fs_time"]
    fc = params["fc"]
    Tmp = params["Tmp"]
    # R = params["R"]
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
    a = 1 - 1 / params["f_resamp"]
    t = np.arange(np.round(channel_time * fs_delay).astype(int))
    theta_hat = -a * 2 * np.pi * fc * t[:, None] / fs_delay

    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
    }

    if params["has_theta_hat"]:
        channel["theta_hat"] = theta_hat
    if params["has_f_resamp"]:
        channel["f_resamp"] = np.array([[params["f_resamp"]]])
    if params["has_theta_hat"] and params["has_f_resamp"]:
        # We set the residual theta_hat to zero
        channel["theta_hat"] = np.zeros(channel["theta_hat"].shape)
        channel["f_resamp"] = np.array([[params["f_resamp"]]])

    fs_time = 40
    array_index = [0]
    unpacked = unpack(fs_time, array_index, channel)

    # import matplotlib.pyplot as plt
    # delay_axis = np.arange(unpacked.shape[0]) / fs_delay
    # time_axis = np.arange(unpacked.shape[2]) / fs_time
    # plt.pcolor(delay_axis * 1e3, time_axis, 20 * np.log10(np.abs(np.squeeze(unpacked[:, 0, :] + 1e-10))).T, vmin=-30, vmax=0)
    # plt.xlabel("Delay [ms]")
    # plt.ylabel("Time [s]")
    # plt.show()
