import pytest
import numpy as np
from uwa_replay import unpack
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
    path_delay_0 = np.concatenate(
        ([0], np.sort(randsamples(np.arange(1, p["Tmp"] * 1e3) / 1e3, p["n_path"] - 1)))
    )
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
        phi_sin = (
            p["fc"] * v_amp * T_ch / (C * n_cycles) * (np.cos(omega_osc * t_delay) - 1)
        )
    else:
        phi_sin = np.zeros_like(t_delay)
    phi_full = phi_const + phi_sin

    # Build h_hat
    L = int(np.ceil(p["fs_delay"] * p["Tmp"] * 1.5))
    has_motion = (v_const != 0) or (v_amp != 0)

    if p["tracking"] == "theta" and has_motion:
        h_hat = np.zeros((N_time, 1, L), dtype=complex)
        for k in range(N_time):
            h_hat[k, 0, :] = place_taps(
                L, path_delay_0 + dtau_snap[k], c_p, p["Tmp"], p["fs_delay"]
            )
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
    p = params
    channel = build_unpack_channel(p)

    fs_time_out = p["fs_delay"] * 0.01
    array_index = [0]
    unpacked = unpack(fs_time_out, array_index, channel, 0.3, 0.3)

    assert np.all(np.isfinite(unpacked)), f"Non-finite values in {p['label']}"
    assert unpacked.ndim == 3
    assert unpacked.shape[1] == 1

    # --- Plot ---
    h_hat_complex = np.array(channel["h_hat"]["real"]) + 1j * np.array(
        channel["h_hat"]["imag"]
    )
    N_time = h_hat_complex.shape[0]
    L = h_hat_complex.shape[2]
    t_snapshots = np.arange(N_time) / p["fs_time"]

    delay_axis = np.arange(unpacked.shape[0]) / p["fs_delay"] * 1e3
    time_axis = np.arange(unpacked.shape[2]) / fs_time_out

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Unpacked
    ax = axes[0]
    im = np.squeeze(np.abs(unpacked[:, 0, :])).T
    im = 20 * np.log10(im + 1e-10)
    ax.imshow(
        im,
        aspect="auto",
        extent=[delay_axis[0], delay_axis[-1], time_axis[-1], time_axis[0]],
        vmin=-30,
        vmax=0,
        interpolation="nearest",
    )
    ax.set_xlabel("Delay [ms]")
    ax.set_ylabel("Time [s]")
    ax.set_title("Unpacked")

    # h_hat input
    ax = axes[1]
    delay_axis_h = np.arange(L) / p["fs_delay"] * 1e3
    im_h = np.abs(h_hat_complex[:, 0, :])
    im_h = 20 * np.log10(im_h + 1e-10)
    ax.imshow(
        im_h,
        aspect="auto",
        extent=[delay_axis_h[0], delay_axis_h[-1], t_snapshots[-1], t_snapshots[0]],
        vmin=-30,
        vmax=0,
        interpolation="nearest",
    )
    ax.set_xlabel("Delay [ms]")
    ax.set_ylabel("Time [s]")
    ax.set_title("h_hat (input)")

    # Speed from phase
    ax = axes[2]
    N_delay = round(p["channel_time"] * p["fs_delay"])
    if "theta_hat" in channel:
        phi_plot = channel["theta_hat"][:, 0]
    elif "phi_hat" in channel:
        phi_plot = channel["phi_hat"][:, 0]
    else:
        phi_plot = np.zeros(N_delay)
    if p.get("has_f_resamp", False) and "f_resamp" in channel:
        phi_plot = phi_plot + (
            (1 / channel["f_resamp"][0, 0] - 1)
            * 2
            * np.pi
            * p["fc"]
            * np.arange(1, len(phi_plot) + 1)
            / p["fs_delay"]
        )
    dphi = np.diff(phi_plot)
    v_inst = -dphi / (1 / p["fs_delay"] * 2 * np.pi * p["fc"]) * C
    t_speed = np.arange(len(v_inst)) / p["fs_delay"]
    ax.plot(t_speed, v_inst)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title(r"Speed (from $\phi$)")
    ax.grid(True)

    fig.suptitle(p["label"])
    fig.tight_layout()
    plt.savefig(f"fig_unpack_{p['label']}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
