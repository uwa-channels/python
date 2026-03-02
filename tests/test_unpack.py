import pytest
import numpy as np
from uwa_replay import unpack

C = 1500


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


def randsamples(population, num):
    rand_index = np.random.permutation(len(population))
    return population[rand_index[:num]]


def place_taps(L, delays, gains, Tmp, fs_delay):
    """Place taps for a single element with out-of-window rejection."""
    h = np.zeros(L, dtype=complex)
    subs = np.round((delays + 0.2 * Tmp) * fs_delay).astype(int)
    valid = (subs >= 0) & (subs < L)
    h[subs[valid]] = gains[valid]
    return h


def build_unpack_channel(p):
    """Build a synthetic single-element channel for unpack testing."""
    # Multipath geometry
    path_delay_0 = np.concatenate(([0], np.sort(randsamples(np.arange(1, p["Tmp"] * 1e3) / 1e3, p["n_path"] - 1))))
    path_delay_0 -= np.min(path_delay_0)

    path_gain = np.exp(-path_delay_0 * p["coeff"] / p["Tmp"])
    c_p = path_gain * np.exp(-1j * 2 * np.pi * p["fc"] * path_delay_0)

    # Motion model
    T_ch = p["channel_time"]
    N_time = round(T_ch * p["fs_time"])
    N_delay = round(T_ch * p["fs_delay"])

    t_snapshots = np.arange(N_time) / p["fs_time"]
    t_delay = (np.arange(1, N_delay + 1)) / p["fs_delay"]

    v_const = p["v_const"]
    v_amp = p["v_amp"]
    n_cycles = p["n_cycles"]

    # Cumulative delay at snapshot times
    dtau_snap = (v_const / C) * t_snapshots
    if n_cycles > 0:
        omega_osc = 2 * np.pi * n_cycles / T_ch
        dtau_snap -= v_amp / (C * omega_osc) * (np.cos(omega_osc * t_snapshots) - 1)

    # Cumulative phase
    phi_const = -2 * np.pi * p["fc"] * (v_const / C) * t_delay
    if n_cycles > 0:
        phi_sin = p["fc"] * v_amp * T_ch / (C * n_cycles) * (np.cos(omega_osc * t_delay) - 1)
    else:
        phi_sin = np.zeros_like(t_delay)
    phi_full = phi_const + phi_sin

    # Build h_hat
    L = int(np.ceil(p["fs_delay"] * p["Tmp"] * 1.5))
    has_motion = (v_const != 0) or (v_amp != 0)

    if p["tracking"] == "theta" and has_motion:
        h_hat = np.zeros((N_time, 1, L), dtype=complex)
        for k in range(N_time):
            h_hat[k, 0, :] = place_taps(L, path_delay_0 + dtau_snap[k], c_p, p["Tmp"], p["fs_delay"])
    else:
        h_hat_static = place_taps(L, path_delay_0, c_p, p["Tmp"], p["fs_delay"])
        h_hat = np.tile(h_hat_static[None, None, :], (N_time, 1, 1))

    # Tracking fields
    if p.get("has_f_resamp", False):
        f_resamp = 1 / (1 + v_const / C)
        phi_tracking = phi_sin
    else:
        phi_tracking = phi_full

    # Assemble channel
    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "params": {
            "fs_delay": np.array([[p["fs_delay"]]]),
            "fs_time": np.array([[p["fs_time"]]]),
            "fc": np.array([[p["fc"]]]),
        },
    }

    phi_2d = phi_tracking[:, None]
    if p["tracking"] == "theta":
        channel["theta_hat"] = phi_2d
    elif p["tracking"] == "phi":
        channel["phi_hat"] = phi_2d
    # tracking == "none": no field added

    if p.get("has_f_resamp", False):
        channel["f_resamp"] = np.array([[f_resamp]])

    return channel


PARAMS = [
    # Static, no tracking
    {
        "label": "static_none",
        "fc": 10e3,
        "fs_delay": 8e3,
        "fs_time": 20,
        "n_path": 8,
        "Tmp": 10e-3,
        "coeff": 1,
        "channel_time": 5,
        "tracking": "none",
        "has_f_resamp": False,
        "v_const": 0,
        "v_amp": 0,
        "n_cycles": 0,
    },
    # Constant drift, theta
    {
        "label": "drift_theta",
        "fc": 10e3,
        "fs_delay": 8e3,
        "fs_time": 100,
        "n_path": 8,
        "Tmp": 10e-3,
        "coeff": 1,
        "channel_time": 5,
        "tracking": "theta",
        "has_f_resamp": False,
        "v_const": -2,
        "v_amp": 0,
        "n_cycles": 0,
    },
    {
        "label": "drift_theta_fast",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "n_path": 8,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "channel_time": 5,
        "tracking": "theta",
        "has_f_resamp": False,
        "v_const": 4,
        "v_amp": 0,
        "n_cycles": 0,
    },
    # Constant drift, phi
    {
        "label": "drift_phi",
        "fc": 10e3,
        "fs_delay": 8e3,
        "fs_time": 20,
        "n_path": 8,
        "Tmp": 10e-3,
        "coeff": 1,
        "channel_time": 5,
        "tracking": "phi",
        "has_f_resamp": False,
        "v_const": -2,
        "v_amp": 0,
        "n_cycles": 0,
    },
    {
        "label": "drift_phi_fast",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "n_path": 8,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "channel_time": 5,
        "tracking": "phi",
        "has_f_resamp": False,
        "v_const": 4,
        "v_amp": 0,
        "n_cycles": 0,
    },
    # f_resamp only
    {
        "label": "f_resamp_only",
        "fc": 15e3,
        "fs_delay": 16e3,
        "fs_time": 20,
        "n_path": 10,
        "Tmp": 20e-3,
        "coeff": 1,
        "channel_time": 5,
        "tracking": "none",
        "has_f_resamp": True,
        "v_const": 2,
        "v_amp": 0,
        "n_cycles": 0,
    },
    # f_resamp + theta
    {
        "label": "f_resamp_theta",
        "fc": 15e3,
        "fs_delay": 16e3,
        "fs_time": 100,
        "n_path": 10,
        "Tmp": 20e-3,
        "coeff": 1,
        "channel_time": 5,
        "tracking": "theta",
        "has_f_resamp": True,
        "v_const": 2,
        "v_amp": 0.3,
        "n_cycles": 3,
    },
    # f_resamp + phi
    {
        "label": "f_resamp_phi",
        "fc": 15e3,
        "fs_delay": 16e3,
        "fs_time": 20,
        "n_path": 10,
        "Tmp": 20e-3,
        "coeff": 1,
        "channel_time": 5,
        "tracking": "phi",
        "has_f_resamp": True,
        "v_const": 2,
        "v_amp": 0.3,
        "n_cycles": 3,
    },
    # Sway, theta
    {
        "label": "sway_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "n_path": 8,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "channel_time": 5,
        "tracking": "theta",
        "has_f_resamp": False,
        "v_const": 0,
        "v_amp": 1.5,
        "n_cycles": 4,
    },
    # Sway, phi
    {
        "label": "sway_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "n_path": 8,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "channel_time": 5,
        "tracking": "phi",
        "has_f_resamp": False,
        "v_const": 0,
        "v_amp": 1.5,
        "n_cycles": 4,
    },
    # Combined, theta
    {
        "label": "combined_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "n_path": 8,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "channel_time": 5,
        "tracking": "theta",
        "has_f_resamp": False,
        "v_const": 3,
        "v_amp": 0.5,
        "n_cycles": 3,
    },
    # Combined, phi
    {
        "label": "combined_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "n_path": 8,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "channel_time": 5,
        "tracking": "phi",
        "has_f_resamp": False,
        "v_const": 3,
        "v_amp": 0.5,
        "n_cycles": 3,
    },
]


@pytest.mark.parametrize("params", PARAMS, ids=[p["label"] for p in PARAMS])
def test_unpack_function(params):
    """Verify unpack produces finite output with correct shape."""
    channel = build_unpack_channel(params)

    fs_time_out = params["fs_delay"] * 0.01
    array_index = [0]
    unpacked = unpack(fs_time_out, array_index, channel, 0.3, 0.3)

    assert np.all(np.isfinite(unpacked)), f"Non-finite values in {params['label']}"
    assert unpacked.ndim == 3
    assert unpacked.shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
