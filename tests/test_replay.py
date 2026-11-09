import pytest
import numpy as np
import scipy.signal as sg
from scipy.interpolate import CubicSpline
from fractions import Fraction
from uwa_channels import replay
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    path_delay_0 = np.concatenate(
        ([0], np.sort(randsamples(np.arange(1, p["Tmp"] * 1e3) / 1e3, p["n_path"] - 1)))
    )[:, None]

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

    # Cumulative phase at fs_delay rate (split into constant and sway parts)
    phi = -2 * np.pi * p["fc"] * (v_const / C) * t_delay
    phi_sin = np.zeros_like(t_delay)
    if n_cycles > 0:
        phi_sin = (
            p["fc"] * v_amp * T_ch / (C * n_cycles) * (np.cos(omega_osc * t_delay) - 1)
        )
        phi = phi + phi_sin

    # Build h_hat
    L = int(np.ceil(p["fs_delay"] * p["Tmp"] * 1.5))
    has_motion = (v_const != 0) or (v_amp != 0)

    if p["tracking"] == "theta" and has_motion:
        h_hat = np.zeros((N_time, p["M"], L), dtype=complex)
        for k in range(N_time):
            h_hat[k, :, :] = place_taps(
                L, p["M"], path_delay_0 + dtau_snap[k], c_p, p["Tmp"], p["fs_delay"]
            )
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

    return channel, path_delay_0, path_gain, phi, has_motion, t_snapshots, t_delay, L


# =========================================================================
#  Test parameters
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

    channel, path_delay_0, path_gain, phi, has_motion, t_snapshots, t_delay, L = (
        build_channel(p)
    )

    # Get phase field for compensation
    if p["tracking"] == "theta":
        phi_field = channel["theta_hat"]
    elif p["tracking"] == "phi":
        phi_field = channel["phi_hat"]
    else:
        phi_field = np.zeros((len(t_delay), p["M"]))

    # Transmit signal
    data_symbols = np.random.randint(2, size=(4095,)) * 2 - 1
    baseband = sg.resample_poly(data_symbols, int(fs / p["R"]), 1)
    passband = np.real(
        baseband * np.exp(2j * np.pi * p["fc"] * np.arange(len(baseband)) / fs)
    )
    input_signal = np.concatenate(
        (np.zeros(int(round(fs / 10))), passband, np.zeros(int(round(fs / 10))))
    )

    # Replay
    r = replay(input_signal, fs, np.arange(p["M"]), channel, start)

    # h_hat for plotting
    h_hat_complex = np.array(channel["h_hat"]["real"]) + 1j * np.array(
        channel["h_hat"]["imag"]
    )
    delay_axis = np.arange(L) / p["fs_delay"] * 1e3

    # Replay time window
    frac_rs = Fraction(p["fs_delay"] / fs).limit_denominator()
    T_baseband = len(sg.resample_poly(baseband, frac_rs.numerator, frac_rs.denominator))
    t_replay_start = (start - 1) / p["fs_delay"]
    t_replay_end = (start + T_baseband + L) / p["fs_delay"]

    # --- Figure: 2 top, 1 bottom ---
    fig = plt.figure(figsize=(12, 8))
    ax_hhat = fig.add_subplot(2, 2, 1)
    ax_speed = fig.add_subplot(2, 2, 2)
    ax_xcor = fig.add_subplot(2, 1, 2)

    # (top-left) h_hat waterfall: delay on x, time on y
    ax_hhat.imshow(
        np.abs(h_hat_complex[:, 0, :]),
        aspect="auto",
        extent=[delay_axis[0], delay_axis[-1], t_snapshots[-1], t_snapshots[0]],
        interpolation="nearest",
    )
    ax_hhat.axhline(t_replay_start, color="r", linestyle="--", linewidth=1.5)
    ax_hhat.axhline(t_replay_end, color="r", linestyle="--", linewidth=1.5)
    ax_hhat.set_xlabel("Delay [ms]")
    ax_hhat.set_ylabel("Time [s]")
    ax_hhat.set_title("|h_hat| (element 0)")

    # (top-right) Speed from phase
    dphi = np.diff(phi_field[:, 0])
    dt = 1 / p["fs_delay"]
    v_inst = -dphi / (dt * 2 * np.pi * p["fc"]) * C
    t_speed = np.arange(len(v_inst)) / p["fs_delay"]
    ax_speed.plot(t_speed, v_inst)
    # A constant speed differs from sample to sample only by rounding error
    # (~1e-12 m/s); a span of at least 1 m/s keeps the axis from zooming in.
    v_mid = (v_inst.max() + v_inst.min()) / 2
    v_half = max(0.55 * (v_inst.max() - v_inst.min()), 0.5)
    ax_speed.set_ylim(v_mid - v_half, v_mid + v_half)
    ax_speed.axvline(t_replay_start, color="r", linestyle="--", linewidth=1.5)
    ax_speed.axvline(t_replay_end, color="r", linestyle="--", linewidth=1.5)
    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title(
        "Instantaneous speed "
        + {"theta": r"(from $\theta$)", "phi": r"(from $\phi$)"}.get(
            p["tracking"], "(no tracking field)"
        )
    )
    ax_speed.grid(True)

    # (bottom) Ground truth stems + Cross-correlation
    baseband_ref = baseband
    sync_ref = None
    max_xcor = 0
    criteria = np.zeros(p["M"], dtype=bool)

    # Plot ground truth as stems first
    for m in range(p["M"]):
        ax_xcor.stem(
            path_delay_0[:, m] * 1e3,
            path_gain[:, m],
            markerfmt="o",
            basefmt=" ",
            linefmt="C%d-" % m,
            label=f"truth el {m}",
        )

    for m in range(p["M"]):
        v_m = r[:, m] * np.exp(-2j * np.pi * p["fc"] * np.arange(len(r)) / fs)

        if has_motion:
            v_m = compensate_doppler(
                v_m, fs, p["fc"], p["fs_delay"], phi_field[:, m], start
            )

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

        win = (lags_shifted >= -0.2 * p["Tmp"] * fs) & (
            lags_shifted <= 1.5 * p["Tmp"] * fs
        )
        xcor_win = xcor_norm[win]
        lags_win = lags_shifted[win]

        min_gain = np.min(np.abs(path_gain[:, m])) * 0.6
        min_sep = np.min(np.diff(np.sort(path_delay_0[:, m]))) * fs * 0.7
        peaks, _ = sg.find_peaks(xcor_win, height=min_gain, distance=int(min_sep))
        peaks = peaks[: p["n_path"]]

        xaxis = lags_win / fs * 1e3
        ax_xcor.plot(xaxis, xcor_win, "--", color=f"C{m}", alpha=0.7)
        if len(peaks) > 0:
            ax_xcor.plot(
                xaxis[peaks], xcor_win[peaks], "x", color=f"C{m}", markersize=10
            )

        # Criterion: gain-weighted delay error against ground truth.
        # Weighting both sides by the TRUE path gains keeps the metric a
        # pure delay-fidelity test. Peak heights from a random binary
        # probe carry sidelobe noise of about 1/sqrt(4095) per path,
        # which is a property of the probe, not of replay; folding the
        # estimated gains into the criterion lets that noise dominate
        # for dense channels.
        n_found = len(peaks)
        if n_found == p["n_path"]:
            est_delays = np.sort(lags_win[peaks] / fs)
            true_delays = np.sort(path_delay_0[:, m])
            true_gains = path_gain[np.argsort(path_delay_0[:, m]), m]

            tol = 2e-4 * p["n_path"]
            criteria[m] = np.sum(np.abs(est_delays - true_delays) * true_gains) < tol

    ax_xcor.set_xlabel("Delay [ms]")
    ax_xcor.set_ylabel("Path gain / |Xcorr|")
    ax_xcor.set_xlim([-0.2 * p["Tmp"] * 1e3, 1.5 * p["Tmp"] * 1e3])
    ax_xcor.set_title("Ground truth + Cross-correlation")
    ax_xcor.legend(fontsize=7, ncol=2)

    result = "PASSED" if np.all(criteria) else "FAILED"
    fig.suptitle(f"{p['label']}: {result}")
    fig.tight_layout()
    plt.savefig(f"test_replay_{p['label']}.png", dpi=150)
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
    input_signal = np.random.randn(1024)
    output = replay(input_signal, fs, np.arange(5), channel)

    assert output.shape[1] == 5  # all channels in h_hat
    assert output.shape[0] > 0
    assert np.isfinite(output).all()


def drift_channel(fc, fs_delay, n_time, L, rate):
    """A single-tap channel whose delay drifts at ``rate`` seconds per second,
    held as phi_hat = -2 pi fc rate t, with fs_time = 20 Hz."""
    fs_time = 20.0
    h = np.zeros((n_time, 1, L), dtype=complex)
    h[:, 0, L // 2] = 1
    t = np.arange(int(n_time / fs_time * fs_delay)) / fs_delay
    return {
        "h_hat": {"real": h.real, "imag": h.imag},
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
        "version": np.array([[1.0]]),
        "phi_hat": (-2 * np.pi * fc * rate * t)[:, None],
    }


@pytest.mark.parametrize("rate", [2e-3, -2e-3], ids=["receding", "closing"])
def test_late_start_keeps_both_ends_of_the_signal(rate):
    """A replay late in a drifting channel must not lose either end of the input.

    phi_hat accumulates over the whole record, so late in a channel whose
    delay drifts steadily its absolute value is large.  replay reinserts the
    drift relative to the starting sample; reinserting the absolute drift
    shifted the output by all of it, which cut off the end of the signal when
    the delay grows (receding) and its beginning when it shrinks (closing).
    The delay here drifts by 2 ms per second, so 15 s in it is 30 ms, against
    1 ms of change across the 0.5 s input.  A burst at each end must survive.
    """
    fs, fc, fs_delay = 96e3, 13e3, 8e3
    channel = drift_channel(fc, fs_delay, 400, 64, rate)

    x = np.zeros(int(0.5 * fs))
    b = int(0.005 * fs)
    burst = np.cos(2 * np.pi * fc * np.arange(b) / fs)
    x[:b] = burst
    x[-b:] = burst

    early = replay(x, fs, [0], channel, start=1000)[:, 0]
    late = replay(x, fs, [0], channel, start=int(15 * fs_delay))[:, 0]

    ratio = np.sum(late**2) / np.sum(early**2)
    assert 0.9 < ratio < 1.1, (
        f"a replay 15 s in carries {ratio:.2f} of the energy of one at the "
        f"start; part of the signal was cut off"
    )
    w = int(0.05 * fs)
    head = np.sum(late[:w] ** 2) / np.sum(late**2)
    tail = np.sum(late[-w:] ** 2) / np.sum(late**2)
    assert head > 0.3, f"the burst at the start of the input is missing ({head:.2f})"
    assert tail > 0.3, f"the burst at the end of the input is missing ({tail:.2f})"


def test_output_length():
    """The output is T + L samples at fs_delay, plus room for the drift.

    T is the input length at fs_delay and L the number of taps.  With phi_hat,
    the output also holds the largest delay the drift reaches in the replay
    window, measured from the starting sample: with a delay growing at a
    constant rate r that is ceil(r * (T + L - 1)) samples, and a shrinking
    delay needs none, nor do theta_hat and no tracking.  The MATLAB toolbox
    returns the same length.
    """
    fs, fc, fs_delay, L = 96e3, 13e3, 8e3, 64
    x = np.random.randn(int(0.5 * fs))
    T = int(np.ceil(x.size * fs_delay / fs))
    up = int(fs / fs_delay)
    rate = 2e-3

    channel = drift_channel(fc, fs_delay, 400, L, rate)
    n = replay(x, fs, [0], channel, start=1000).shape[0]
    assert (
        n == (T + L + int(np.ceil(rate * (T + L - 1)))) * up
    ), "phi_hat, growing delay"

    channel = drift_channel(fc, fs_delay, 400, L, -rate)
    assert (
        replay(x, fs, [0], channel, start=1000).shape[0] == (T + L) * up
    ), "phi_hat, shrinking delay"

    channel["theta_hat"] = channel.pop("phi_hat")
    assert replay(x, fs, [0], channel, start=1000).shape[0] == (T + L) * up, "theta_hat"

    del channel["theta_hat"]
    assert (
        replay(x, fs, [0], channel, start=1000).shape[0] == (T + L) * up
    ), "no tracking"


def test_random_start_stays_in_range():
    """Without a start, replay draws one at random.

    With the input nearly as long as the channel allows, the valid starts are
    a handful of samples, so a draw past the last one reads beyond phi_hat or
    h_hat and errors, and so would one before the first.
    """
    fs, fc, fs_delay, fs_time, L = 96e3, 13e3, 8e3, 20.0, 64
    n_time = 40  # 2 s of channel
    channel = drift_channel(fc, fs_delay, n_time, L, 2e-3)
    T_max = int(n_time / fs_time * fs_delay)
    T = T_max - L - 5  # 5 valid starts
    x = np.random.randn(int(T * fs / fs_delay))
    for _ in range(200):
        assert np.isfinite(replay(x, fs, [0], channel)).all()


@pytest.mark.parametrize("nsd", [1, 2], ids=["fullBand", "halfBand"])
def test_replay_power(nsd):
    """Two checks for the array_index API and the h_hat normalization:

    1. Isolation: replaying through a single hydrophone must equal the
       corresponding column of a replay through all hydrophones (same
       start). This holds only because output normalization was removed,
       so each channel is processed independently of the selected set.
    2. Power: a channel file estimated at N_s = fs_delay/R samples per
       symbol holds a response confined to the signal band |f| <= R/2,
       and its h_hat is normalized so that the energy over the taps and
       all M hydrophones, averaged over time, is M/N_s.  A unit-power
       passband signal of bandwidth R replayed through all M hydrophones
       over the full length of the channel must then give output powers
       that sum to M.  A subset of the hydrophones need not give a
       proportional share of M, since their powers differ, and a short
       signal need not give M, since it sees only part of the fading.
    """
    fs = FS

    fc = 12e3
    R = 4e3
    fs_delay = nsd * R
    fs_time = 20
    M = 4
    n_path = 8
    Tmp = 15e-3
    coeff = 1.5
    d = 0.5
    T_ch = 5

    # Multipath geometry, with a Doppler shift on each path.  The shifts
    # move the fades of the frequency response across the band during the
    # channel, as fading does, so that a replay over the full length
    # averages them.  A static channel keeps its fades, and its output
    # power depends on where they fall.
    path_delay_0 = np.concatenate(
        ([0], np.sort(randsamples(np.arange(1, Tmp * 1e3) / 1e3, n_path - 1)))
    )[:, None]
    incremental_delay = np.arange(M)[None, :] * d / C
    path_delay_0 = path_delay_0 + incremental_delay
    path_delay_0 -= np.min(path_delay_0)
    path_gain = np.exp(-path_delay_0 * coeff / Tmp)
    c_p = path_gain * np.exp(-1j * 2 * np.pi * fc * path_delay_0)
    f_doppler = 4 * (np.random.rand(n_path, 1) - 0.5)  # within +/- 2 Hz

    L = int(np.ceil(fs_delay * Tmp * 1.5))
    N_time = round(T_ch * fs_time)
    h_hat = np.zeros((N_time, M, L), dtype=complex)
    for k in range(N_time):
        rotation = np.exp(2j * np.pi * f_doppler * k / fs_time)
        h_hat[k] = place_taps(L, M, path_delay_0, c_p * rotation, Tmp, fs_delay)

    # Confine the response to the signal band, as an estimate is.  The
    # cutoff R/2 is 1/N_s of the Nyquist frequency fs_delay/2.
    if nsd > 1:
        n_fir = 64
        h_hat = sg.lfilter(
            sg.firwin(n_fir + 1, 1 / nsd),
            1,
            np.concatenate((h_hat, np.zeros((N_time, M, n_fir))), axis=2),
            axis=2,
        )
    L = h_hat.shape[2]

    # Normalize as the channel files are: energy M/N_s
    E_avg = np.sum(np.abs(h_hat) ** 2) / N_time
    h_hat = np.sqrt(M / nsd / E_avg) * h_hat

    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
        "version": np.array([[1.0]]),
    }

    # Unit-power passband probe in the signal band, as long as the channel.
    # The probe is 0.9 R wide: with N_s = 1 the band fills the whole delay
    # band, and the resampler in replay attenuates its edges, which would
    # cost a probe of width R about 6% of its power.  replay needs T + L
    # samples at fs_delay after the start, and the channel holds
    # N_time / fs_time * fs_delay of them.
    B = 0.9 * R
    T_max = N_time / fs_time * fs_delay
    n_sym = int(np.floor((T_max - L - 2) / fs_delay * B))
    data_symbols = np.sign(np.random.randn(n_sym)) + 1j * np.sign(
        np.random.randn(n_sym)
    )
    ratio = Fraction(fs / B).limit_denominator(1000)
    baseband = sg.resample_poly(data_symbols, ratio.numerator, ratio.denominator)
    passband = np.real(
        baseband * np.exp(2j * np.pi * fc * np.arange(len(baseband)) / fs)
    )
    passband = passband / np.sqrt(np.mean(passband**2))

    # Replay one channel, and all channels, from the first sample
    target_ch = 2
    r_single = replay(passband, fs, [target_ch], channel, 0)
    r_all = replay(passband, fs, np.arange(M), channel, 0)

    # Check 1: isolation
    n = min(r_single.shape[0], r_all.shape[0])
    ref = r_all[:n, target_ch]
    err = np.max(np.abs(r_single[:n, 0] - ref)) / np.max(np.abs(ref))
    assert err < 1e-10, (
        f"Single-channel replay does not match column {target_ch} of the "
        f"full replay (relative err = {err:.2e})."
    )

    # Check 2: the output powers over all M hydrophones sum to M.  The
    # output energy is divided by the input length, so the tail that the
    # channel adds to the output does not lower the power.
    P = np.sum(r_all**2, axis=0) / len(passband)
    print(
        f"test_replay_power: N_s = {nsd}, sum of output powers / M = {np.sum(P) / M:.3f}"
    )
    assert abs(np.sum(P) / M - 1) < 0.05, (
        f"Output powers sum to {np.sum(P):.3f}, not M = {M}, with N_s = {nsd} "
        f"and h_hat normalized to M/N_s."
    )


# ---------------------------------------------------------------------------
# f_resamp with one factor per hydrophone (Watermark)
# ---------------------------------------------------------------------------


def static_array_channel(taps_per_element, fs_delay, fs_time, fc, n_time, L):
    """A static channel; column m of taps_per_element is element m's CIR."""
    taps = np.asarray(taps_per_element, dtype=complex)  # (L, M)
    h = np.tile(taps.T[None, :, :], (n_time, 1, 1))  # (T, M, L)
    return {
        "h_hat": {"real": h.real, "imag": h.imag},
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
        "version": np.array([[1.0]]),
    }


def test_vector_f_resamp():
    """Each column of a vector-f_resamp replay is that element replayed alone.

    f_resamp may hold one factor per hydrophone.  Column k must then equal a
    replay through hydrophone array_index[k] alone with its factor as a
    scalar, zero-padded to the longest column.  A vector of equal factors must
    give the same output as the scalar, and array_index must pick the factors
    of the hydrophones it selects, in its order.
    """
    fs, fc, fs_delay, fs_time, L, n_time, M = 48e3, 12e3, 10e3, 20.0, 64, 100, 3
    taps = np.zeros((L, M), dtype=complex)
    taps[9, 0] = 1
    taps[[11, 29], 1] = [0.8, 0.4j]
    taps[[7, 20, 39], 2] = [1, -0.5, 0.3j]
    channel = static_array_channel(taps, fs_delay, fs_time, fc, n_time, L)
    t = np.arange(int(n_time / fs_time * fs_delay)) / fs_delay
    channel["phi_hat"] = np.tile((-2 * np.pi * fc * 1e-4 * t)[:, None], (1, M))

    n = int(0.5 * fs)
    x = np.cos(2 * np.pi * fc * np.arange(n) / fs) * np.hanning(n)
    f_resamp = 1 / (1 + np.array([2e-3, -1e-3, 4e-3]))
    channel["f_resamp"] = f_resamp[:, None]

    for array_index in ([0, 1, 2], [2, 0]):
        r = replay(x, fs, array_index, channel, start=START)
        lengths = []
        for k, m in enumerate(array_index):
            single = static_array_channel(
                taps[:, [m]], fs_delay, fs_time, fc, n_time, L
            )
            single["phi_hat"] = channel["phi_hat"][:, [m]]
            single["f_resamp"] = np.array([[f_resamp[m]]])
            r_m = replay(x, fs, [0], single, start=START)[:, 0]
            lengths.append(len(r_m))
            err = np.max(np.abs(r[: len(r_m), k] - r_m)) / np.max(np.abs(r_m))
            assert err < 1e-10, (
                f"column {k} (hydrophone {m}) of replay(..., {array_index}, ...) "
                f"differs from hydrophone {m} replayed alone with f_resamp = "
                f"{f_resamp[m]:.6f} (relative err = {err:.2e})"
            )
            assert np.all(r[len(r_m) :, k] == 0), f"column {k} is not zero-padded"
        assert r.shape[0] == max(lengths), "output is not as long as its longest column"
        assert len(set(lengths)) > 1, "the factors are too close to change the lengths"

    # Equal factors as a row vector give the scalar result.
    channel["f_resamp"] = np.array([[f_resamp[0]]])
    r_scalar = replay(x, fs, [0, 1, 2], channel, start=START)
    channel["f_resamp"] = np.full((1, M), f_resamp[0])
    r_vector = replay(x, fs, [0, 1, 2], channel, start=START)
    assert r_vector.shape == r_scalar.shape
    assert np.max(np.abs(r_vector - r_scalar)) / np.max(np.abs(r_scalar)) < 1e-12


def burst_peaks(r):
    """Sample indices of the two burst peaks in r, one in each half of it."""
    env = np.abs(sg.hilbert(r))
    half = len(env) // 2
    return np.array([np.argmax(env[:half]), half + np.argmax(env[half:])])


def test_vector_f_resamp_shared_cir():
    """Same CIR on every hydrophone, different f_resamp: each column stretches.

    Any difference between the outputs comes from f_resamp alone.  replay
    resamples column m by f_resamp[m], which stretches it in time by that
    factor: a column is f_resamp[m] times as long as without it, and two
    events D seconds apart come out f_resamp[m] * D apart.  The input is two
    short bursts at fc, 0.4 s apart.  The figure shows each whole column with
    its end marked, and the second burst with where it falls without f_resamp
    (dashed) and how far f_resamp moves it.
    """
    fs, fc, fs_delay, fs_time, L, n_time, M = 48e3, 12e3, 8e3, 20.0, 64, 100, 3
    taps = np.zeros(L, dtype=complex)
    taps[[9, 17, 29]] = [1, 0.5 - 0.3j, 0.25j]
    channel = static_array_channel(
        np.tile(taps[:, None], (1, M)), fs_delay, fs_time, fc, n_time, L
    )

    b = int(2e-3 * fs)
    burst = np.cos(2 * np.pi * fc * np.arange(b) / fs) * np.hanning(b)
    x = np.zeros(int(0.5 * fs))
    n1, D = int(0.05 * fs), 0.4
    x[n1 : n1 + b] = burst
    x[n1 + int(D * fs) : n1 + int(D * fs) + b] = burst

    r0 = replay(x, fs, [0], channel, start=START)[:, 0]
    q0, len0 = burst_peaks(r0), len(r0)

    excess = np.array([1e-2, -1e-2, -3e-2])  # 1 / f_resamp - 1
    f_resamp = 1 / (1 + excess)
    channel["f_resamp"] = f_resamp[None, :]
    r = replay(x, fs, np.arange(M), channel, start=START)

    fig, axes = plt.subplots(M, 2, figsize=(14, 8))
    lengths = []
    for m in range(M):
        single = static_array_channel(taps[:, None], fs_delay, fs_time, fc, n_time, L)
        single["f_resamp"] = np.array([[f_resamp[m]]])
        lengths.append(len(replay(x, fs, [0], single, start=START)))
        q = burst_peaks(r[: lengths[m], m])

        assert abs(lengths[m] - len0 * f_resamp[m]) < 2, (
            f"hydrophone {m}: the column is {lengths[m]} samples long, not "
            f"f_resamp * {len0} = {len0 * f_resamp[m]:.1f}"
        )
        assert abs(np.diff(q)[0] - np.diff(q0)[0] * f_resamp[m]) < 2, (
            f"hydrophone {m}: the bursts are {np.diff(q)[0]} samples apart, not "
            f"f_resamp * {np.diff(q0)[0]} = {np.diff(q0)[0] * f_resamp[m]:.1f}"
        )

        env = np.abs(sg.hilbert(r[:, m]))
        env /= env.max()
        t = np.arange(len(env)) / fs

        ax = axes[m, 0]
        ax.plot(t, env, lw=0.8)
        ax.axvline(lengths[m] / fs, color="r", lw=1.5)
        ax.axvline(len0 / fs, color="0.5", ls="--", lw=1.5)
        ax.set_xlim(0, r.shape[0] / fs * 1.02)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("|r|")
        ax.grid(True)
        ax.set_title(
            f"Hydrophone {m}: f_resamp = 1/(1{excess[m]:+.0e}), "
            f"length {lengths[m]} ({len0} without)"
        )

        ax = axes[m, 1]
        shift = (q[1] - q0[1]) / fs * 1e3
        ax.plot(t * 1e3, env, lw=0.8)
        ax.axvline(q0[1] / fs * 1e3, color="0.5", ls="--", lw=1.5)
        ax.axvline(q[1] / fs * 1e3, color="r", lw=0.8)
        ax.plot(np.array([q0[1], q[1]]) / fs * 1e3, [1.05, 1.05], "r-", lw=2)
        ax.text(
            (q0[1] + q[1]) / 2 / fs * 1e3,
            0.85,
            f"{shift:+.2f} ms",
            color="r",
            ha="center",
            fontweight="bold",
        )
        ax.set_xlim(q0[1] / fs * 1e3 - 8, q0[1] / fs * 1e3 + 20)
        ax.set_ylim(0, 1.15)
        ax.grid(True)
        ax.set_title(f"Second burst moves by {shift:+.2f} ms")
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [ms]")
    fig.suptitle(
        "Same CIR, different f_resamp: each column is stretched by its own "
        "factor (dashed: without f_resamp)"
    )
    fig.tight_layout()
    plt.savefig("test_replay_vector_f_resamp.png", dpi=150)
    plt.close(fig)

    assert len(set(lengths)) == M, "the columns should all have different lengths"
    assert r.shape[0] == max(lengths), "output is not as long as its longest column"


@pytest.mark.parametrize(
    "f_resamp", [[1.0, 1.0], [1.0, -1.0, 1.0], [1.0, np.nan, 1.0], [0.0]]
)
def test_f_resamp_checked(f_resamp):
    """A vector f_resamp needs one positive, finite factor per hydrophone."""
    channel = drift_channel(13e3, 8e3, 40, 64, 1e-4)
    channel["h_hat"] = {k: np.tile(v, (1, 3, 1)) for k, v in channel["h_hat"].items()}
    channel["phi_hat"] = np.tile(channel["phi_hat"], (1, 3))
    channel["f_resamp"] = np.array([f_resamp])
    with pytest.raises(ValueError, match="f_resamp"):
        replay(np.random.randn(int(0.1 * 96e3)), 96e3, [0], channel, start=1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
