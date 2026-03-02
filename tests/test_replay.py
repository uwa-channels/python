import pytest
import numpy as np
import scipy.signal as sg
from scipy.interpolate import CubicSpline
from fractions import Fraction
from uwa_replay import replay
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 1500
FS = 48e3
START = 1000


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


def randsamples(population, num):
    rand_index = np.random.permutation(len(population))
    return population[rand_index[:num]]


def place_taps(L, M, delays, gains, Tmp, fs_delay):
    """Place taps with out-of-window rejection."""
    h = np.zeros((M, L), dtype=complex)
    subs = np.round((delays + 0.2 * Tmp) * fs_delay).astype(int)
    valid = (subs >= 0) & (subs < L)
    for m in range(M):
        v = valid[:, m]
        h[m, subs[v, m]] = gains[v, m]
    return h


def compensate_doppler(v_in, fs, fc, fs_delay, phi_field, start):
    """Remove Doppler using the actual phase field."""
    N = len(v_in)
    t_rx = np.arange(N) / fs
    t_phi = np.arange(len(phi_field)) / fs_delay
    t_abs = t_rx + (start - 1) / fs_delay

    phi_rx = CubicSpline(t_phi, phi_field)(t_abs)
    phi_rx -= phi_rx[0]

    dtau = -phi_rx / (2 * np.pi * fc)

    v_comp = v_in * np.exp(-1j * phi_rx)
    v_out = CubicSpline(t_rx, v_comp)(t_rx + dtau)
    return v_out


def build_channel(p):
    """Build a synthetic channel struct from test parameters."""
    # Multipath geometry
    path_delay_0 = np.concatenate(([0], np.sort(randsamples(np.arange(1, p["Tmp"] * 1e3) / 1e3, p["n_path"] - 1))))[
        :, None
    ]

    incremental_delay = np.arange(p["M"])[None, :] * p["d"] / C
    path_delay_0 = path_delay_0 + incremental_delay
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

    # Cumulative phase at fs_delay rate
    phi = -2 * np.pi * p["fc"] * (v_const / C) * t_delay
    if n_cycles > 0:
        phi += p["fc"] * v_amp * T_ch / (C * n_cycles) * (np.cos(omega_osc * t_delay) - 1)

    # Build h_hat
    L = int(np.ceil(p["fs_delay"] * p["Tmp"] * 1.5))
    has_motion = (v_const != 0) or (v_amp != 0)

    if p["tracking"] == "theta" and has_motion:
        h_hat = np.zeros((N_time, p["M"], L), dtype=complex)
        for k in range(N_time):
            h_hat[k, :, :] = place_taps(L, p["M"], path_delay_0 + dtau_snap[k], c_p, p["Tmp"], p["fs_delay"])
    else:
        h_hat_static = place_taps(L, p["M"], path_delay_0, c_p, p["Tmp"], p["fs_delay"])
        h_hat = np.tile(h_hat_static[None, :, :], (N_time, 1, 1))

    # Assemble channel dict
    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "params": {
            "fs_delay": np.array([[p["fs_delay"]]]),
            "fs_time": np.array([[p["fs_time"]]]),
            "fc": np.array([[p["fc"]]]),
        },
        "version": np.array([[1.0]]),
    }

    phi_2d = np.tile(phi[:, None], (1, p["M"]))
    if p["tracking"] == "theta":
        channel["theta_hat"] = phi_2d
    elif p["tracking"] == "phi":
        channel["phi_hat"] = phi_2d

    return channel, path_delay_0, path_gain, phi, has_motion, t_snapshots, L


# =========================================================================
#  Test parameters — matches MATLAB testReplay.m exactly
# =========================================================================

PARAMS = [
    # Static
    {
        "label": "static_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "M": 3,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": 0,
        "v_amp": 0,
        "n_cycles": 0,
    },
    {
        "label": "static_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "M": 3,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": 0,
        "v_amp": 0,
        "n_cycles": 0,
    },
    # Low speed AUV (1 m/s) + mild sway
    {
        "label": "low_speed_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "M": 3,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": 1,
        "v_amp": 0.2,
        "n_cycles": 2,
    },
    {
        "label": "low_speed_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 20,
        "M": 3,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": 1,
        "v_amp": 0.2,
        "n_cycles": 2,
    },
    # Moderate speed (3 m/s) + platform sway
    {
        "label": "moderate_speed_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 2,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": 3,
        "v_amp": 0.5,
        "n_cycles": 3,
    },
    {
        "label": "moderate_speed_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 2,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": 3,
        "v_amp": 0.5,
        "n_cycles": 3,
    },
    # High speed (5 m/s) + strong sway
    {
        "label": "high_speed_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 2,
        "n_path": 6,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": 5,
        "v_amp": 1.0,
        "n_cycles": 4,
    },
    {
        "label": "high_speed_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 2,
        "n_path": 6,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": 5,
        "v_amp": 1.0,
        "n_cycles": 4,
    },
    # Negative drift (closing) + sway
    {
        "label": "closing_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 2,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": -3,
        "v_amp": 0.5,
        "n_cycles": 3,
    },
    {
        "label": "closing_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 2,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": -3,
        "v_amp": 0.5,
        "n_cycles": 3,
    },
    # Pure sway, no drift
    {
        "label": "sway_only_theta",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 3,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": 0,
        "v_amp": 1.5,
        "n_cycles": 5,
    },
    {
        "label": "sway_only_phi",
        "fc": 12e3,
        "fs_delay": 10e3,
        "fs_time": 100,
        "M": 3,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": 0,
        "v_amp": 1.5,
        "n_cycles": 5,
    },
    # Single element, high speed
    {
        "label": "single_elem_theta",
        "fc": 12e3,
        "fs_delay": 8e3,
        "fs_time": 100,
        "M": 1,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "theta",
        "v_const": 4,
        "v_amp": 1.2,
        "n_cycles": 3,
    },
    {
        "label": "single_elem_phi",
        "fc": 12e3,
        "fs_delay": 8e3,
        "fs_time": 100,
        "M": 1,
        "n_path": 8,
        "R": 4e3,
        "Tmp": 15e-3,
        "coeff": 1.5,
        "d": 0.5,
        "channel_time": 5,
        "tracking": "phi",
        "v_const": 4,
        "v_amp": 1.2,
        "n_cycles": 3,
    },
]


@pytest.mark.parametrize("params", PARAMS, ids=[p["label"] for p in PARAMS])
def test_replay_function(params):
    p = params
    fs = FS
    start = START

    channel, path_delay_0, path_gain, phi, has_motion, t_snapshots, L = build_channel(p)

    # Get phase field for compensation
    if p["tracking"] == "theta":
        phi_field = channel["theta_hat"]
    else:
        phi_field = channel["phi_hat"]

    # Transmit signal
    data_symbols = np.random.randint(2, size=(4095,)) * 2 - 1
    baseband = sg.resample_poly(data_symbols, int(fs / p["R"]), 1)
    passband = np.real(baseband * np.exp(2j * np.pi * p["fc"] * np.arange(len(baseband)) / fs))
    input_signal = np.concatenate((np.zeros(int(round(fs / 10))), passband, np.zeros(int(round(fs / 10)))))

    # Replay
    r = replay(input_signal, fs, list(range(p["M"])), channel, start)

    # h_hat for plotting
    h_hat_complex = np.array(channel["h_hat"]["real"]) + 1j * np.array(channel["h_hat"]["imag"])
    delay_axis = np.arange(L) / p["fs_delay"] * 1e3

    # Replay time window
    frac_rs = Fraction(p["fs_delay"] / fs).limit_denominator()
    T_baseband = len(sg.resample_poly(baseband, frac_rs.numerator, frac_rs.denominator))
    t_replay_start = (start - 1) / p["fs_delay"]
    t_replay_end = (start + T_baseband + L) / p["fs_delay"]

    # --- Figure: 2x2 layout ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) h_hat waterfall: delay on x, time on y
    ax = axes[0, 0]
    ax.imshow(
        np.abs(h_hat_complex[:, 0, :]),
        aspect="auto",
        extent=[delay_axis[0], delay_axis[-1], t_snapshots[-1], t_snapshots[0]],
        interpolation="nearest",
    )
    ax.axhline(t_replay_start, color="r", linestyle="--", linewidth=1.5)
    ax.axhline(t_replay_end, color="r", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Delay [ms]")
    ax.set_ylabel("Time [s]")
    ax.set_title("|h_hat| (element 0)")

    # (0,1) Speed from phase
    ax = axes[0, 1]
    dphi = np.diff(phi_field[:, 0])
    dt = 1 / p["fs_delay"]
    v_inst = -dphi / (dt * 2 * np.pi * p["fc"]) * C
    t_speed = np.arange(len(v_inst)) / p["fs_delay"]
    ax.plot(t_speed, v_inst)
    ax.axvline(t_replay_start, color="r", linestyle="--", linewidth=1.5)
    ax.axvline(t_replay_end, color="r", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title(r"Instantaneous speed (from $\phi$)")
    ax.grid(True)

    # (1,0) Ground truth
    h1 = axes[1, 0]
    for m in range(p["M"]):
        h1.stem(
            path_delay_0[:, m] * 1e3,
            path_gain[:, m],
            markerfmt="o",
            basefmt=" ",
            label=f"el {m}",
        )
    h1.set_xlabel("Delay [ms]")
    h1.set_ylabel("Path gain")
    h1.set_title("Ground truth")
    h1.set_xlim([-0.2 * p["Tmp"] * 1e3, 1.5 * p["Tmp"] * 1e3])

    # (1,1) Cross-correlation
    h2 = axes[1, 1]

    baseband_ref = baseband
    sync_ref = None
    max_xcor = 0
    criteria = np.zeros(p["M"], dtype=bool)

    for m in range(p["M"]):
        v_m = r[:, m] * np.exp(-2j * np.pi * p["fc"] * np.arange(len(r)) / fs)

        if has_motion:
            v_m = compensate_doppler(v_m, fs, p["fc"], p["fs_delay"], phi_field[:, m], start)

        xcor = sg.correlate(v_m, baseband_ref, mode="full")
        lags = np.arange(len(xcor)) - (len(baseband_ref) - 1)

        xcor = xcor[lags > 0]
        lags = lags[lags > 0]

        sync_idx = np.argmax(np.abs(xcor))
        if m == 0:
            sync_ref = sync_idx
            max_xcor = np.max(np.abs(xcor))

        lags_shifted = lags - sync_ref
        xcor_norm = np.abs(xcor) / max_xcor

        win = (lags_shifted >= -0.2 * p["Tmp"] * fs) & (lags_shifted <= 1.5 * p["Tmp"] * fs)
        xcor_win = xcor_norm[win]
        lags_win = lags_shifted[win]

        min_gain = np.min(np.abs(path_gain[:, m])) * 0.6
        min_sep = np.min(np.diff(np.sort(path_delay_0[:, m]))) * fs * 0.7
        peaks, _ = sg.find_peaks(xcor_win, height=min_gain, distance=int(min_sep))
        peaks = peaks[: p["n_path"]]

        xaxis = lags_win / fs * 1e3
        h2.plot(xaxis, xcor_win)
        if len(peaks) > 0:
            h2.plot(xaxis[peaks], xcor_win[peaks], "x", markersize=10)

        n_found = len(peaks)
        if n_found > 0:
            est_delays = lags_win[peaks] / fs
            est_gains = xcor_win[peaks]
            idx_e = np.argsort(est_delays)
            idx_t = np.argsort(path_delay_0[:, m])

            n_compare = min(n_found, p["n_path"])
            tol = 2e-4 * p["n_path"]
            criteria[m] = (
                np.abs(
                    np.sum(est_delays[idx_e[:n_compare]] * est_gains[idx_e[:n_compare]])
                    - np.sum(path_delay_0[idx_t[:n_compare], m] * path_gain[idx_t[:n_compare], m])
                )
                < tol
            )

    h2.set_xlabel("Delay [ms]")
    h2.set_ylabel("|Xcorr|")
    h2.set_xlim([-0.2 * p["Tmp"] * 1e3, 1.5 * p["Tmp"] * 1e3])
    h2.set_title("Cross-correlation")

    result = "PASSED" if np.all(criteria) else "FAILED"
    fig.suptitle(f"{p['label']}: {result}")
    fig.tight_layout()
    plt.savefig(f"fig_{p['label']}.png", dpi=150)
    plt.close(fig)

    assert np.all(criteria), f"Peak delay mismatch for case: {p['label']}"


def test_replay_basic():
    """Basic smoke test: output is finite and correctly shaped."""
    fs = 96e3
    channel = {
        "h_hat": {
            "real": np.random.randn(400, 5, 100),
            "imag": np.random.randn(400, 5, 100),
        },
        "params": {
            "fs_delay": np.array([[8e3]]),
            "fs_time": np.array([[20.0]]),
            "fc": np.array([[13e3]]),
        },
        "version": np.array([[1.0]]),
    }
    array_index = [0, 1]
    input_signal = np.random.randn(1024)
    output = replay(input_signal, fs, array_index, channel)

    assert output.shape[1] == len(array_index)
    assert output.shape[0] > 0
    assert np.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
