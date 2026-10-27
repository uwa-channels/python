"""End-to-end tests against the released library files on Zenodo.

The other suites build a channel and a noise dict in memory, hand them to the
package, and check what comes back.  That misses a whole class of defect: the
fixture is written by the same conventions the reader assumes, so a reader that
disagrees with the *files* -- an axis in the wrong order, a MAT flavour it
cannot open -- still passes.  Exactly that shipped once.

So this suite starts from the files themselves.  It downloads a channel and the
noise measured on the same array from the version-specific record
10.5281/zenodo.21287414 (see ``conftest.py`` for the cache and the opt-in),
then walks the path a user walks: ``load_channel`` and ``load_noise``, a probe
signal through ``replay``, ``noisegen`` on top, and a check that what comes out
is the channel that went in.

What is actually asserted about the replayed signal, since "it ran" is not a
test:

* the arrival lands where the file says it should.  The matched-filter peak is
  compared against the guard delay plus the strongest tap of ``h_hat`` minus
  the delay drift ``phi_hat / (2*pi*fc)`` that ``replay`` re-inserts, and has
  to agree to better than two samples of ``fs_delay``.  That number comes from
  three separate places in the file, so it cannot be right by accident.
* the delay profile has the right shape.  The measured power-delay profile is
  compared, in dB over the taps within 25 dB of the peak, against the profile
  computed from ``h_hat`` over the snapshots the probe spans.
* the noise has the spatial covariance ``beta`` prescribes, and matches it far
  better than it matches the transposed convention -- the shape of the bug that
  prompted this suite, tested here on the shipped coefficients rather than on a
  fixture written to agree.

Authors: Zhengnan Li, Claude (Opus 5)
Email  : uwa-channels@ofdm.link
License: MIT

Revision history:
  - Sep. 15, 2026: Initial release.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.signal as sg
import h5py

from conftest import CHANNEL_FILE, NOISE_FILE
from uwa_channels import replay, noisegen, load_channel, load_noise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.zenodo

STEM = Path(CHANNEL_FILE).stem

# Probe: a random BPSK sequence at the bandwidth the noise file was measured
# over.  8192 symbols is 0.7 s on the black array, long enough for ~26 dB of
# processing gain and short enough that the channel does not change much.
N_SYM = 8192
GUARD = 0.02  # s of silence before and after the probe
N_BLOCK = 8  # matched-filter blocks averaged incoherently
ARRAY_INDEX = [0, 3, 7]

# Tolerances.  Arrival time is a hard geometric prediction, so it gets a tight
# bound; the profile shape is a correlation over a handful of taps and the
# concentration is diluted by the correlation sidelobe floor, so those get
# loose ones.  Measured values on black.mat, over three start times and three
# hydrophones: arrival error < 0.7 samples, dB correlation 0.72-0.91,
# concentration 0.84-0.91.
ARRIVAL_TOL = 2.0  # samples of fs_delay
PROFILE_CORR_MIN = 0.6
CONCENTRATION_MIN = 0.7


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


# ---------------------------------------------------------------------------
# The released files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def channel(channel_path):
    handle = load_channel(channel_path)
    yield handle
    if isinstance(handle, h5py.File):
        handle.close()


@pytest.fixture(scope="session")
def noise(noise_path):
    handle = load_noise(noise_path)
    yield handle
    if isinstance(handle, h5py.File):
        handle.close()


def channel_params(channel):
    """fs_delay, fs_time, fc of an open channel."""
    p = channel["params"]
    return (
        float(p["fs_delay"][0, 0]),
        float(p["fs_time"][0, 0]),
        float(p["fc"][0, 0]),
    )


def noise_params(noise):
    """Fs, R, fc, alpha of an open noise file."""
    return tuple(
        float(np.asarray(noise[k]).ravel()[0]) for k in ("Fs", "R", "fc", "alpha")
    )


def beta_matlab(noise):
    """``beta`` in the stored MATLAB layout (M, M, K), whichever loader read it.

    h5py reverses every axis, so a MATLAB ``[M, M, K]`` array arrives as
    ``(K, M, M)``; ``scipy.io.loadmat`` hands back the stored layout.  This is
    the same axis-order fix-up ``noisegen`` applies internally, repeated here
    so the test states the convention it checks instead of importing it.
    """
    b = np.asarray(noise["beta"])
    assert b.ndim == 3, f"beta should be 3-D, got {b.shape}"
    if b.shape[0] != b.shape[1]:
        b = np.transpose(b, (2, 1, 0))
    return b


def coherence(cov):
    d = np.sqrt(np.diag(cov))
    return np.abs(cov / np.outer(d, d))


# ---------------------------------------------------------------------------
# The probe, the replay, and the matched filter
# ---------------------------------------------------------------------------


def make_probe(fs, fc, R, n_sym=N_SYM, seed=0):
    """A guarded passband BPSK probe, and its baseband at ``fs``."""
    symbols = np.random.default_rng(seed).choice([-1.0, 1.0], size=n_sym)
    ratio = Fraction(fs / R).limit_denominator()
    baseband = sg.resample_poly(symbols, ratio.numerator, ratio.denominator)
    passband = np.real(
        baseband * np.exp(2j * np.pi * fc * np.arange(len(baseband)) / fs)
    )
    guard = int(round(GUARD * fs))
    return np.concatenate((np.zeros(guard), passband, np.zeros(guard))), baseband


def to_baseband(y, fs, fc, fs_delay):
    """Down-convert a replayed passband signal and resample it to fs_delay."""
    ratio = Fraction(fs_delay / fs).limit_denominator()
    v = y * np.exp(-2j * np.pi * fc * np.arange(y.shape[0])[:, None] / fs)
    return sg.resample_poly(v, ratio.numerator, ratio.denominator, axis=0)


def measured_pdp(rx, tx, n_block=N_BLOCK):
    """Power-delay profile of ``rx``, by block-averaged matched filtering.

    One correlation against the whole probe would put the sidelobe floor around
    -15 dB, which buries the weak arrivals.  Correlating against ``n_block``
    disjoint pieces instead gives each its own sidelobe pattern, so averaging
    the magnitudes squared pushes the floor down by roughly 10*log10(n_block)
    while staying immune to the phase drift across the probe.

    Index ``d`` of the result is delay ``d`` samples after the start of the
    transmission, the same origin the returned profile of ``true_pdp`` uses.
    """
    length = len(tx) // n_block
    total = None
    for b in range(n_block):
        piece = tx[b * length : (b + 1) * length]
        power = np.abs(sg.correlate(rx, piece, mode="valid")) ** 2
        # The peak for block b sits b*length later than for block 0; line the
        # blocks up before averaging.
        aligned = power[b * length : b * length + len(power) - (n_block - 1) * length]
        total = aligned if total is None else total + aligned
    return total / n_block


def true_pdp(channel, m, start, n_samples):
    """Power-delay profile of ``h_hat[m]``, averaged over the snapshots spanned."""
    fs_delay, fs_time, _ = channel_params(channel)
    first = int(start / fs_delay * fs_time)
    last = int((start + n_samples) / fs_delay * fs_time) + 1
    block = channel["h_hat"][first : last + 1]
    h = block["real"][:, m, :] + 1j * block["imag"][:, m, :]
    return np.mean(np.abs(h) ** 2, axis=0)


def drift_samples(channel, m, start, n_samples):
    """Mean delay drift over the probe, in samples of fs_delay.

    ``replay`` evaluates the output at ``t + phi_hat/(2*pi*fc)``, so a positive
    drift advances the arrival.  A theta_hat channel carries no delay drift.
    """
    if "phi_hat" not in channel:
        return 0.0
    fs_delay, _, fc = channel_params(channel)
    phi = np.asarray(channel["phi_hat"][start : start + n_samples, m])
    return float(np.mean(phi) / (2 * np.pi * fc) * fs_delay)


def subsample_peak(p, index=None):
    """Peak position of ``p`` to sub-sample accuracy, by parabolic fit."""
    i = int(np.argmax(p)) if index is None else int(index)
    if i <= 0 or i >= len(p) - 1:
        return float(i)
    a, b, c = p[i - 1], p[i], p[i + 1]
    curvature = a - 2 * b + c
    return float(i) if curvature == 0 else i + 0.5 * (a - c) / curvature


@pytest.fixture(scope="session")
def replayed(channel, noise):
    """Replay one probe through the released channel, a quarter of the way in.

    Session-scoped: ``replay`` reads all of ``h_hat``, so the tests share one
    run rather than repeating it.
    """
    fs_delay, fs_time, fc = channel_params(channel)
    fs, R, _, _ = noise_params(noise)
    n_snapshot = channel["h_hat"].shape[0]

    x, baseband = make_probe(fs, fc, R)
    start = int(0.25 * n_snapshot / fs_time * fs_delay)

    np.random.seed(1994)
    y = replay(x, fs, ARRAY_INDEX, channel, start=start)

    ratio = Fraction(fs_delay / fs).limit_denominator()
    tx = sg.resample_poly(baseband, ratio.numerator, ratio.denominator)
    tx = tx / np.linalg.norm(tx)

    return {
        "x": x,
        "y": y,
        "v": to_baseband(y, fs, fc, fs_delay),
        "tx": tx,
        "start": start,
        "fs": fs,
        "R": R,
        # Delay from the first transmitted sample to the first output sample,
        # in samples of fs_delay: the guard, resampled.
        "guard": int(round(GUARD * fs)) * fs_delay / fs,
    }


# ---------------------------------------------------------------------------
# What the files contain
# ---------------------------------------------------------------------------


def test_channel_file_matches_the_published_format(channel):
    """The released channel is laid out the way the README says it is."""
    assert isinstance(channel, h5py.File), f"{CHANNEL_FILE} should be a -v7.3 file"
    assert float(np.asarray(channel["version"]).ravel()[0]) >= 1.0

    fs_delay, fs_time, fc = channel_params(channel)
    assert fs_delay > 0 and fs_time > 0 and fc > 0
    assert fs_time < fs_delay, "the time axis is the slow one"

    # h5py reverses the MATLAB axes, so the stored (K, M, T) arrives as
    # (T, M, K): snapshots first, hydrophones second, delay taps last.
    n_snapshot, M, K = channel["h_hat"].shape
    assert M < n_snapshot and M < K, (
        f"h_hat is {channel['h_hat'].shape}; the hydrophone axis should be the "
        f"middle one, so this file looks transposed"
    )
    assert channel["h_hat"].dtype.names == ("real", "imag")

    tracking = [k for k in ("theta_hat", "phi_hat") if k in channel]
    assert len(tracking) == 1, f"expected exactly one tracking field, got {tracking}"

    # The phase field runs at fs_delay for the whole span of h_hat, one column
    # per hydrophone.  Getting this axis backwards is the failure this guards.
    phase = channel[tracking[0]]
    assert phase.shape[1] == M, (
        f"{tracking[0]} is {phase.shape}; its second axis should be the "
        f"{M} hydrophones of h_hat"
    )
    expected = int(np.ceil(n_snapshot * fs_delay / fs_time))
    assert phase.shape[0] == pytest.approx(expected, rel=1e-3)

    snapshot = channel["h_hat"][:32]
    assert np.isfinite(snapshot["real"]).all() and np.isfinite(snapshot["imag"]).all()
    assert np.isfinite(phase[:100000]).all()


def test_noise_file_matches_the_published_format(noise, channel):
    """The released noise struct is laid out the way noisegen expects."""
    assert isinstance(noise, h5py.File), f"{NOISE_FILE} should be a -v7.3 file"
    assert float(np.asarray(noise["version"]).ravel()[0]) >= 1.0

    Fs, R, fc, alpha = noise_params(noise)
    assert 0 < alpha <= 2
    assert 0 < R and 0 < fc - R / 2, "the passband should start above DC"
    assert fc + R / 2 < Fs / 2, "the passband should fit below Nyquist"

    beta = beta_matlab(noise)
    M = beta.shape[0]
    assert beta.shape[1] == M, f"beta should be square in (i, j), got {beta.shape}"
    assert M == channel["h_hat"].shape[1], (
        f"{NOISE_FILE} has {M} elements and {CHANNEL_FILE} has "
        f"{channel['h_hat'].shape[1]}; they should be the same array"
    )
    assert np.isfinite(beta).all()

    # beta is normalized so the per-channel pseudo-powers sum to the element
    # count; each element keeps its measured power relative to the others.
    power = np.sum(np.abs(beta) ** alpha, axis=(1, 2)) ** (2.0 / alpha)
    assert np.sum(power) == pytest.approx(M, rel=1e-3)

    # And beta is asymmetric in (i, j), which is what makes the axis order
    # observable at all -- a symmetric beta would read the same transposed.
    asymmetry = np.max(np.abs(beta - np.transpose(beta, (1, 0, 2)))) / np.max(
        np.abs(beta)
    )
    assert asymmetry > 0.01, "beta is nearly symmetric; this file cannot catch a swap"


# ---------------------------------------------------------------------------
# Reading the released files back as MAT v5
# ---------------------------------------------------------------------------


def test_released_noise_survives_a_v5_round_trip(noise, tmp_path):
    """Written without -v7.3 and read back, the shipped beta generates the same noise.

    ``test_io`` checks this on a fixture; here the coefficients are the shipped
    ones, so the two loaders are reconciled on data neither of them wrote.
    """
    Fs, R, fc, alpha = noise_params(noise)
    beta = beta_matlab(noise)
    M = beta.shape[0]

    v5 = tmp_path / "noise_v5.mat"
    sio.savemat(
        v5,
        {
            "Fs": Fs,
            "R": R,
            "alpha": alpha,
            "fc": fc,
            "beta": beta,
            "version": float(np.asarray(noise["version"]).ravel()[0]),
        },
    )
    assert not h5py.is_hdf5(str(v5)), "savemat should have written MAT v5 here"

    index = list(range(M))
    np.random.seed(101)
    from_v73 = noisegen((1 << 15, M), Fs, index, noise)
    np.random.seed(101)
    from_v5 = noisegen((1 << 15, M), Fs, index, load_noise(v5))
    assert np.allclose(from_v5, from_v73)


def test_released_channel_survives_a_v5_round_trip(channel, noise, tmp_path):
    """An excerpt of the released channel replays identically from either flavour.

    A short excerpt keeps the file small; all three axes stay different lengths
    so a transposed read cannot slip through on a coincidence of shape.
    """
    fs_delay, fs_time, fc = channel_params(channel)
    fs, R, _, _ = noise_params(noise)
    n_snapshot, M, K = channel["h_hat"].shape

    first = n_snapshot // 2
    span = 64
    assert len({span, M, K}) == 3, "excerpt axes must stay distinguishable"

    block = channel["h_hat"][first : first + span]
    excerpt = block["real"] + 1j * block["imag"]  # (span, M, K), h5py order
    tracking = "phi_hat" if "phi_hat" in channel else "theta_hat"
    n_phase = int(np.ceil(span * fs_delay / fs_time))
    offset = int(first * fs_delay / fs_time)
    phase = np.asarray(channel[tracking][offset : offset + n_phase])

    params = np.zeros((1, 1), dtype=[("fs_delay", "O"), ("fs_time", "O"), ("fc", "O")])
    params["fs_delay"][0, 0] = np.array([[fs_delay]])
    params["fs_time"][0, 0] = np.array([[fs_time]])
    params["fc"][0, 0] = np.array([[fc]])

    v5 = tmp_path / "channel_v5.mat"
    sio.savemat(
        v5,
        {
            # Back to the stored MATLAB layout: (K, M, T) and (T, M) -> (M, T).
            "h_hat": np.transpose(excerpt, (2, 1, 0)),
            tracking: phase.T,
            "params": params,
            "version": 1.0,
        },
    )
    assert not h5py.is_hdf5(str(v5))

    from_v5 = load_channel(v5)
    assert from_v5["h_hat"].shape == excerpt.shape
    assert np.allclose(np.asarray(from_v5["h_hat"]["real"]), excerpt.real)
    assert np.allclose(np.asarray(from_v5["h_hat"]["imag"]), excerpt.imag)
    assert np.allclose(np.asarray(from_v5[tracking]), phase)

    # And the excerpt replays the same through either reader.
    in_memory = {
        "h_hat": {"real": excerpt.real, "imag": excerpt.imag},
        tracking: phase,
        "params": {
            "fs_delay": np.array([[fs_delay]]),
            "fs_time": np.array([[fs_time]]),
            "fc": np.array([[fc]]),
        },
        "version": np.array([[1.0]]),
    }
    x, _ = make_probe(fs, fc, R, n_sym=512)
    index = [0, M - 1]
    assert np.allclose(
        replay(x, fs, index, from_v5, start=0),
        replay(x, fs, index, in_memory, start=0),
    )


# ---------------------------------------------------------------------------
# noisegen on the released coefficients
# ---------------------------------------------------------------------------


def test_noisegen_reproduces_the_released_spatial_covariance(noise):
    """Generated noise has the covariance beta prescribes, not its transpose.

    For Gaussian noise the mixing w_i = sum_{j,k} beta_ij(k) z_{j,n+k} of unit
    white drivers gives covariance sum_k B_k B_k^T.  Reading beta with the
    hydrophone axes swapped would give sum_k B_k^T B_k instead, which is a
    different matrix for the shipped coefficients -- so this both checks the
    covariance and pins down the axis order that produces it.
    """
    Fs, _, _, alpha = noise_params(noise)
    assert alpha == 2, "this file is not the Gaussian one; adjust the test"

    beta = beta_matlab(noise)
    M, _, K = beta.shape
    published = sum(beta[:, :, k] @ beta[:, :, k].T for k in range(K))
    swapped = sum(beta[:, :, k].T @ beta[:, :, k] for k in range(K))

    np.random.seed(1994)
    w = noisegen((1 << 18, M), Fs, list(range(M)), noise)
    assert np.isfinite(w).all()

    # Per-channel variance, against the diagonal of the prescribed covariance.
    assert np.allclose(np.var(w, axis=0), np.diag(published), rtol=0.05)

    sample = coherence(np.cov(w.T))
    to_published = np.sqrt(np.mean((sample - coherence(published)) ** 2))
    to_swapped = np.sqrt(np.mean((sample - coherence(swapped)) ** 2))

    assert to_published < 0.02, (
        f"sample coherence is {to_published:.4f} rms from the coherence beta "
        f"prescribes"
    )
    assert to_published < 0.25 * to_swapped, (
        f"generated noise matches the swapped mixing convention (rms "
        f"{to_swapped:.4f}) about as well as the published one "
        f"(rms {to_published:.4f})"
    )


def test_noisegen_is_bandpass_at_the_released_fc_and_R(noise):
    """The generated noise occupies the band the file was measured over."""
    Fs, R, fc, _ = noise_params(noise)
    beta = beta_matlab(noise)
    M = beta.shape[0]

    np.random.seed(1994)
    w = noisegen((1 << 18, M), Fs, list(range(M)), noise)
    f, psd = sg.welch(w, fs=Fs, nperseg=4096, axis=0)

    in_band = (f >= fc - R / 2) & (f <= fc + R / 2)
    far_out = (f < fc - R) | (f > fc + R)
    fraction = psd[in_band].sum() / psd.sum()
    rejection = 10 * np.log10(psd[in_band].mean() / psd[far_out].mean())

    assert fraction > 0.7, f"only {fraction:.2f} of the noise power is in band"
    # The skirts are real measured ambient noise, not a filter design, so the
    # rejection outside fc +/- R is a modest 19 dB on black_noise rather than
    # the tens of dB a synthetic bandpass beta would give.
    assert rejection > 15, f"out-of-band rejection is only {rejection:.1f} dB"

    # Resampling to another rate must not move the band.  Any rate that still
    # clears the top of the passband will do; three quarters of Fs is a real
    # resampling (3/4) rather than a pass-through.
    other = 0.75 * Fs
    assert other / 2 > fc + R / 2
    w2 = noisegen((1 << 17, len(ARRAY_INDEX)), other, ARRAY_INDEX, noise)
    f2, psd2 = sg.welch(w2, fs=other, nperseg=4096, axis=0)
    band2 = (f2 >= fc - R / 2) & (f2 <= fc + R / 2)
    assert psd2[band2].sum() / psd2.sum() > 0.7

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(f / 1e3, psd[:, 0], label=f"{Fs / 1e3:.0f} kHz")
    ax.semilogy(f2 / 1e3, psd2[:, 0], label=f"{other / 1e3:.0f} kHz", alpha=0.8)
    ax.axvspan((fc - R / 2) / 1e3, (fc + R / 2) / 1e3, color="0.85", zorder=0)
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD")
    ax.set_title(
        f"{NOISE_FILE}: noisegen spectrum, in-band fraction {fraction:.2f}, "
        f"rejection {rejection:.0f} dB"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"test_zenodo_{STEM}_noise.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# replay on the released channel
# ---------------------------------------------------------------------------


def test_replay_reproduces_the_released_delay_profile(channel, replayed):
    """The replayed arrival lands where h_hat and phi_hat together say it should.

    For each hydrophone the matched-filter peak is compared against

        guard + argmax_tap(h_hat) - mean(phi_hat) / (2*pi*fc) * fs_delay,

    and the measured power-delay profile against the one computed from h_hat
    over the snapshots the probe spans.
    """
    fs_delay, _, _ = channel_params(channel)
    K = channel["h_hat"].shape[2]
    v, tx, start, guard = (replayed[k] for k in ("v", "tx", "start", "guard"))

    assert v.shape[1] == len(ARRAY_INDEX)
    assert np.isfinite(v).all()
    assert "f_resamp" not in channel, (
        f"{CHANNEL_FILE} resamples in passband, which moves the arrival by a "
        f"factor this test does not model"
    )

    fig, axes = plt.subplots(len(ARRAY_INDEX), 1, figsize=(8, 9), sharex=True)
    report = []

    for column, (ax, m) in enumerate(zip(np.atleast_1d(axes), ARRAY_INDEX)):
        power = measured_pdp(v[:, column], tx)
        peak_to_median = 10 * np.log10(power.max() / np.median(power))
        assert peak_to_median > 10, (
            f"hydrophone {m}: matched-filter peak is only {peak_to_median:.1f} dB "
            f"above the floor; the probe did not come through"
        )

        truth = true_pdp(channel, m, start, len(tx))
        drift = drift_samples(channel, m, start, len(tx))
        predicted = guard + subsample_peak(truth) - drift
        measured = subsample_peak(power)
        assert abs(measured - predicted) < ARRIVAL_TOL, (
            f"hydrophone {m}: arrival at sample {measured:.2f}, expected "
            f"{predicted:.2f} (guard {guard:.1f} + tap "
            f"{subsample_peak(truth):.2f} - drift {drift:.2f})"
        )

        # Compare the profiles over the taps the file says carry the energy.
        origin = int(round(guard - drift))
        assert origin >= 0, (
            f"a drift of {drift:.1f} samples outruns the {guard:.0f}-sample "
            f"guard; lengthen GUARD"
        )
        measured_profile = power[origin : origin + K]
        measured_profile = measured_profile / measured_profile.max()
        true_profile = truth / truth.max()
        support = true_profile > 10 ** (-2.5)
        assert support.sum() >= 5, "the channel has too few taps to compare"

        correlation = np.corrcoef(
            10 * np.log10(measured_profile[support] + 1e-12),
            10 * np.log10(true_profile[support] + 1e-12),
        )[0, 1]
        concentration = measured_profile[support].sum() / measured_profile.sum()
        assert correlation > PROFILE_CORR_MIN, (
            f"hydrophone {m}: measured delay profile correlates {correlation:.2f} "
            f"with the one in {CHANNEL_FILE}"
        )
        assert concentration > CONCENTRATION_MIN, (
            f"hydrophone {m}: only {concentration:.2f} of the measured energy "
            f"falls on the taps h_hat says carry it"
        )
        report.append((m, measured - predicted, correlation, concentration))

        delay_ms = np.arange(len(measured_profile)) / fs_delay * 1e3
        ax.plot(delay_ms, 10 * np.log10(measured_profile + 1e-12), label="replayed")
        ax.plot(
            delay_ms,
            10 * np.log10(true_profile[: len(measured_profile)] + 1e-12),
            "--",
            label="h_hat",
        )
        ax.set_ylim(-45, 3)
        ax.set_ylabel("Power [dB]")
        ax.set_title(
            f"hydrophone {m}: arrival error {measured - predicted:+.2f} samples, "
            f"profile correlation {correlation:.2f}",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    np.atleast_1d(axes)[-1].set_xlabel("Delay [ms]")
    fig.suptitle(f"{CHANNEL_FILE}: replayed vs. stored delay profile")
    fig.tight_layout()
    fig.savefig(f"test_zenodo_{STEM}_replay.png", dpi=150)
    plt.close(fig)

    print(
        "\n".join(
            f"  hydrophone {m}: arrival error {err:+.2f} samples, "
            f"profile correlation {corr:.2f}, concentration {conc:.2f}"
            for m, err, corr, conc in report
        )
    )


def test_replay_isolates_the_requested_hydrophones(channel, noise, replayed):
    """Replaying one hydrophone gives the same column as replaying all three."""
    fs, R, _, _ = noise_params(noise)
    _, _, fc = channel_params(channel)
    x, _ = make_probe(fs, fc, R, n_sym=1024)
    start = replayed["start"]

    together = replay(x, fs, ARRAY_INDEX, channel, start=start)
    for column, m in enumerate(ARRAY_INDEX):
        alone = replay(x, fs, [m], channel, start=start)
        assert np.allclose(alone[:, 0], together[:, column]), (
            f"hydrophone {m} replays differently on its own than as part of "
            f"{ARRAY_INDEX}"
        )


def test_quick_start_runs_on_the_released_files(channel, noise, replayed):
    """The README's replay-plus-noise recipe, on the files it tells you to download."""
    fs = replayed["fs"]
    y = replayed["y"]

    np.random.seed(2026)
    w = noisegen(y.shape, fs, ARRAY_INDEX, noise)
    assert w.shape == y.shape
    assert np.isfinite(w).all()

    scale = 0.05
    r = y + scale * w
    assert np.isfinite(r).all()

    # Both terms are order one: h_hat carries the channel's own gain and beta
    # is normalized so the powers of all the file's elements sum to the
    # element count, so a unit-variance probe comes back at a sane level
    # rather than needing a fudge factor.  Each element's noise power is the
    # one its row of beta sets, not unity, so the three selected here need
    # not sum to three.
    signal = np.var(y, axis=0)
    assert np.all(signal > 1e-3) and np.all(signal < 1e3), f"replay power {signal}"
    beta = np.asarray(noise["beta"])
    if beta.shape[0] != beta.shape[1]:
        beta = np.transpose(beta, (2, 1, 0))
    expected = np.sum(beta[ARRAY_INDEX] ** 2, axis=(1, 2))
    assert np.allclose(np.var(w, axis=0), expected, rtol=0.25), (
        f"noise power {np.var(w, axis=0)} should follow beta, {expected}"
    )

    snr = 10 * np.log10(signal / (scale**2 * np.var(w, axis=0)))
    expected = 10 * np.log10(np.var(r - scale * w, axis=0) / np.var(scale * w, axis=0))
    assert np.allclose(snr, expected, atol=0.1)
    print(f"  SNR at {scale:g} x noise: {np.round(snr, 1)} dB")

# [EOF]
