import pytest
import numpy as np
from uwa_channels import unpack
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

    # Assemble channel
    channel = {
        "h_hat": {"real": np.real(h_hat), "imag": np.imag(h_hat)},
        "params": {
            "fs_delay": np.array([[p["fs_delay"]]]),
            "fs_time": np.array([[p["fs_time"]]]),
            "fc": np.array([[p["fc"]]]),
        },
        "version": np.array([[1.0]]),
    }

    phi_2d = phi_full[:, None]
    if p["tracking"] == "theta":
        channel["theta_hat"] = phi_2d
    elif p["tracking"] == "phi":
        channel["phi_hat"] = phi_2d
    # tracking == "none": no field added

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
        "v_const": 4,
        "v_amp": 0,
        "n_cycles": 0,
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
    dphi = np.diff(phi_plot)
    v_inst = -dphi / (1 / p["fs_delay"] * 2 * np.pi * p["fc"]) * C
    t_speed = np.arange(len(v_inst)) / p["fs_delay"]
    ax.plot(t_speed, v_inst)
    # A constant speed differs from sample to sample only by rounding error
    # (~1e-12 m/s); a span of at least 1 m/s keeps the axis from zooming in.
    v_mid = (v_inst.max() + v_inst.min()) / 2
    v_half = max(0.55 * (v_inst.max() - v_inst.min()), 0.5)
    ax.set_ylim(v_mid - v_half, v_mid + v_half)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title(
        "Speed "
        + {"theta": r"(from $\theta$)", "phi": r"(from $\phi$)"}.get(
            p["tracking"], "(no tracking field)"
        )
    )
    ax.grid(True)

    fig.suptitle(p["label"])
    fig.tight_layout()
    plt.savefig(f"test_unpack_{p['label']}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The f_resamp phase ramp
#
# The parameterized test above draws pictures; these assert.  A ramp built on
# the wrong origin -- starting one sample in rather than at zero -- puts a
# constant phase rotation, and through the drift a constant delay offset, on
# every unpacked tap, and nothing above would notice.  It is what
# uwa-channels/matlab shipped until Sep. 16, 2026.
# ---------------------------------------------------------------------------


def static_channel(taps, n_time, fc, fs_delay, fs_time):
    """A channel whose every snapshot is the same tap vector."""
    h = np.tile(np.asarray(taps, dtype=complex), (n_time, 1, 1))  # (T, 1, K)
    return {
        "h_hat": {"real": h.real, "imag": h.imag},
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
        "version": np.array([[1.0]]),
    }


def test_f_resamp_ramp_starts_at_zero():
    """Adding f_resamp must not move the first output sample.

    ``unpack`` interpolates the trajectory from ``t_orig = arange(N_phi) /
    fs_delay``, so sample *n* sits at ``n / fs_delay`` and the ramp is zero at
    t = 0.  Both runs below share one resampled ``h_hat``, so the resampling
    transients cancel exactly and the only thing left at the first sample is
    what f_resamp contributes there, which is nothing.
    """
    fc, fs_delay, fs_time, fs_out = 15e3, 16e3, 100.0, 160.0
    excess = 1e-3  # 1 / f_resamp - 1
    taps = np.zeros(128, dtype=complex)
    taps[[5, 19, 44, 77]] = [1, 0.6 - 0.3j, -0.4 + 0.2j, 0.15j]

    channel = static_channel(taps, 200, fc, fs_delay, fs_time)
    u0 = unpack(fs_out, [0], channel, 0.3, 0.3)
    channel["f_resamp"] = np.array([[1 / (1 + excess)]])
    u1 = unpack(fs_out, [0], channel, 0.3, 0.3)

    first0, first1 = u0[:, 0, 0], u1[:, 0, 0]
    scale = np.linalg.norm(first0)
    assert scale > 0, "the test channel unpacked to nothing; the fixture is wrong"

    moved = np.linalg.norm(first1 - first0) / scale
    # What a ramp built one sample in would have done, as the yardstick.
    wrong = abs(np.exp(1j * excess * 2 * np.pi * fc / fs_delay) - 1)
    assert moved < wrong / 1e3, (
        f"the f_resamp ramp does not start at zero: the first output sample "
        f"moves by {moved:.2e} of its norm when f_resamp is added, against "
        f"{wrong:.2e} for a ramp built one sample in"
    )


def test_f_resamp_ramp_rate():
    """...and it must then advance at (1/f_resamp - 1) * 2 pi fc per second.

    Measured on the sum over the delay axis, which is the zero-frequency point
    of the delay response and so is untouched by the delay drift f_resamp also
    applies.  ``excess`` is small enough that the drift stays well inside one
    sample over the record, so no energy leaves the delay window.
    """
    fc, fs_delay, fs_time, fs_out = 15e3, 16e3, 100.0, 160.0
    excess = 2e-6
    taps = np.zeros(128, dtype=complex)
    taps[[21, 40, 63]] = [1, 0.5 - 0.2j, 0.25 + 0.1j]

    channel = static_channel(taps, 500, fc, fs_delay, fs_time)
    u0 = unpack(fs_out, [0], channel, 0.3, 0.3)
    channel["f_resamp"] = np.array([[1 / (1 + excess)]])
    u1 = unpack(fs_out, [0], channel, 0.3, 0.3)

    g0, g1 = u0[:, 0, :].sum(axis=0), u1[:, 0, :].sum(axis=0)
    keep = np.abs(g0) > 0.1 * np.abs(g0).max()  # skip the resample ramp-up
    t = np.arange(len(g0)) / fs_out

    measured = np.angle(g1[keep] / g0[keep])
    predicted = excess * 2 * np.pi * fc * t[keep]
    residual = np.angle(np.exp(1j * (measured - predicted)))  # wrap-safe

    assert np.abs(residual).max() < 1e-3, (
        f"the f_resamp phase ramp advances at the wrong rate: worst residual "
        f"{np.abs(residual).max():.2e} rad against a total ramp of "
        f"{predicted[-1]:.3f} rad"
    )


def test_phi_hat_takes_precedence_over_theta_hat():
    """A channel carrying both phi_hat and theta_hat unpacks as with phi_hat alone.

    est saves its one phase trajectory as phi_hat with delay tracking and as
    theta_hat without, so phi_hat is already the whole phase and adding
    theta_hat would count the phase twice.  replay uses phi_hat alone too.
    """
    fc, fs_delay, fs_time, fs_out = 15e3, 16e3, 100.0, 160.0
    n_time = 200
    taps = np.zeros(128, dtype=complex)
    taps[[21, 40, 63]] = [1, 0.5 - 0.2j, 0.25 + 0.1j]

    channel = static_channel(taps, n_time, fc, fs_delay, fs_time)
    t = np.arange(int(np.ceil(n_time * fs_delay / fs_time))) / fs_delay
    channel["phi_hat"] = (-2 * np.pi * fc * 1e-5 * t)[:, None]
    u_phi = unpack(fs_out, [0], channel, 0.3, 0.3)
    channel["theta_hat"] = (0.7 * np.sin(2 * np.pi * 0.5 * t))[:, None]
    u_both = unpack(fs_out, [0], channel, 0.3, 0.3)

    assert np.array_equal(
        u_both, u_phi
    ), "theta_hat changed the unpacked channel although phi_hat is present"


# ---------------------------------------------------------------------------
# f_resamp with one factor per hydrophone (Watermark), and input checks
# ---------------------------------------------------------------------------


def test_vector_f_resamp():
    """Same CIR on every hydrophone, different f_resamp: each slides its own way.

    All three hydrophones share one static CIR and differ only in f_resamp,
    so any difference between them comes from f_resamp alone.  Each
    hydrophone must unpack as it does alone with its factor as a scalar,
    array_index must pick the factors of the hydrophones it selects, and each
    first arrival must end the record where its factor puts it.  A factor
    f_resamp = 1/(1 + e) moves every arrival by -e*t, so over T seconds the
    CIR slides by -e*T; the figure marks that slide and the predicted track.
    """
    fc, fs_delay, fs_time, fs_out = 15e3, 16e3, 100.0, 160.0
    L, n_time, M, buf = 128, 200, 3, 0.3
    taps = np.zeros(L, dtype=complex)
    taps[[39, 54, 69]] = [1, 0.6 - 0.3j, 0.3 + 0.2j]
    h = np.tile(taps[None, None, :], (n_time, M, 1))
    channel = static_channel(taps, n_time, fc, fs_delay, fs_time)
    channel["h_hat"] = {"real": h.real, "imag": h.imag}
    excess = np.array([5e-4, -5e-4, -1.5e-3])  # 1 / f_resamp - 1
    f_resamp = 1 / (1 + excess)
    channel["f_resamp"] = f_resamp[:, None]

    for array_index in ([0, 1, 2], [2, 0]):
        u = unpack(fs_out, array_index, channel, buf, buf)
        for k, m in enumerate(array_index):
            single = static_channel(taps, n_time, fc, fs_delay, fs_time)
            single["f_resamp"] = np.array([[f_resamp[m]]])
            u_m = unpack(fs_out, [0], single, buf, buf)
            err = np.max(np.abs(u[:, [k], :] - u_m)) / np.max(np.abs(u_m))
            assert err < 1e-12, (
                f"hydrophone {m} in unpack(..., {array_index}, ...) differs from "
                f"it unpacked alone with f_resamp = {f_resamp[m]:.6f} "
                f"(relative err = {err:.2e})"
            )

    u = unpack(fs_out, [0, 1, 2], channel, buf, buf)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert (
            np.max(np.abs(u[:, a, :] - u[:, b, :])) > 0.1
        ), f"hydrophones {a} and {b} unpack alike although their f_resamp differ"

    n_left = int(np.ceil(L * buf))
    delay_axis = (np.arange(u.shape[0]) - n_left) / fs_delay
    time_axis = np.arange(u.shape[2]) / fs_out
    T = time_axis[-1]
    tau0 = 39 / fs_delay
    for m in range(M):
        expected = tau0 - excess[m] * T
        window = np.abs(delay_axis - expected) < 4 / fs_delay
        i = np.argmax(np.abs(u[:, m, -1]) * window)
        assert abs(delay_axis[i] - expected) < 1 / fs_delay, (
            f"hydrophone {m}: the first arrival ends at {delay_axis[i] * 1e3:.3f} "
            f"ms, not at {expected * 1e3:.3f} ms"
        )

    fig, axes = plt.subplots(1, M, figsize=(15, 5))
    for m, ax in enumerate(axes):
        im = 20 * np.log10(np.abs(u[:, m, :]).T + 1e-10)
        ax.imshow(
            im,
            aspect="auto",
            extent=[
                delay_axis[0] * 1e3,
                delay_axis[-1] * 1e3,
                time_axis[-1],
                time_axis[0],
            ],
            vmin=-30,
            vmax=0,
            interpolation="nearest",
        )
        slide = -excess[m] * T
        ax.plot((tau0 - excess[m] * time_axis) * 1e3, time_axis, "w--", lw=1)
        ax.annotate(
            "",
            xy=((tau0 + slide) * 1e3, 0.92 * T),
            xytext=(tau0 * 1e3, 0.92 * T),
            arrowprops=dict(arrowstyle="->", color="w", lw=2),
        )
        ax.text(
            (tau0 + slide / 2) * 1e3,
            0.86 * T,
            f"{slide * 1e3:+.2f} ms",
            color="w",
            ha="center",
            fontweight="bold",
        )
        ax.set_xlabel("Delay [ms]")
        ax.set_ylabel("Time [s]")
        ax.set_title(f"Hydrophone {m}: f_resamp = 1/(1{excess[m]:+.1e})")
    fig.suptitle(
        f"Same CIR, different f_resamp: arrivals slide by -(1/f_resamp - 1) T "
        f"over T = {T:.2f} s"
    )
    fig.tight_layout()
    plt.savefig("test_unpack_vector_f_resamp.png", dpi=150)
    plt.close(fig)


def three_element_channel():
    taps = np.zeros(64, dtype=complex)
    taps[10] = 1
    channel = static_channel(taps, 50, 12e3, 8e3, 100.0)
    channel["h_hat"] = {k: np.tile(v, (1, 3, 1)) for k, v in channel["h_hat"].items()}
    return channel


def test_valid_inputs_pass():
    channel = three_element_channel()
    assert np.all(np.isfinite(unpack(100.0, [0, 1, 2], channel)))
    assert np.all(np.isfinite(unpack(100, np.array([2, 0]), channel, 0, 0.5)))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"version": 0.9}, "version"),
        ({"fs": 0}, "fs"),
        ({"fs": -100.0}, "fs"),
        ({"fs": np.nan}, "fs"),
        ({"fs": np.inf}, "fs"),
        ({"fs": [100.0, 200.0]}, "fs"),
        ({"fs": 100j}, "fs"),
        ({"array_index": []}, "array_index"),
        ({"array_index": [-1]}, "array_index"),
        ({"array_index": [3]}, "array_index"),
        ({"array_index": [0, 0]}, "array_index"),
        ({"array_index": [0.5]}, "array_index"),
        ({"buffer_left": -0.1}, "buffers"),
        ({"buffer_left": np.nan}, "buffers"),
        ({"buffer_left": np.inf}, "buffers"),
        ({"buffer_left": [0.1, 0.2]}, "buffers"),
        ({"buffer_right": -0.1}, "buffers"),
        ({"buffer_right": np.nan}, "buffers"),
        ({"buffer_right": np.inf}, "buffers"),
        ({"buffer_right": [0.1, 0.2]}, "buffers"),
        ({"f_resamp": [[1.0, 1.0]]}, "f_resamp"),
        ({"f_resamp": [[1.0, -1.0, 1.0]]}, "f_resamp"),
        ({"f_resamp": [[1.0, np.nan, 1.0]]}, "f_resamp"),
        ({"f_resamp": [[0.0]]}, "f_resamp"),
    ],
)
def test_invalid_inputs(kwargs, match):
    """validate_inputs rejects each bad input with a ValueError that names it."""
    channel = three_element_channel()
    if "version" in kwargs:
        channel["version"] = np.array([[kwargs["version"]]])
    if "f_resamp" in kwargs:
        channel["f_resamp"] = np.array(kwargs["f_resamp"])
    args = {"fs": 100.0, "array_index": [0], "buffer_left": 0.1, "buffer_right": 0.1}
    args.update({k: v for k, v in kwargs.items() if k in args})
    with pytest.raises(ValueError, match=match):
        unpack(
            args["fs"],
            args["array_index"],
            channel,
            args["buffer_left"],
            args["buffer_right"],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
