"""
Unit tests for noisegen.py

Tests three noise generation modes:
  Option 1: Independent pink Gaussian noise (17 dB/decade).
  Option 2: Colored, spatially-correlated Gaussian noise
            (alpha = 2, beta mixing coefficients).
  Option 3: Impulsive (alpha-stable) noise
            (alpha < 2, beta mixing coefficients).

Options 2 and 3 share a unified noise dict with fields:
  Fs, R, alpha, beta, fc, rms_power, version.
The beta mixing coefficients encode both the bandpass spectral shaping
and the per-channel power scaling: they are normalized so each channel
carries unit pseudo-power (2*c^2, the alpha-stable scale measure that
reduces to the variance when alpha = 2), i.e. the per-channel
pseudo-powers summed over channels equal the channel count.

The synthetic noise dict is modeled on real experimental data
(12-element ULA, 65 mixing taps, Fs=39062.5 Hz, fc=13 kHz,
R~4.9 kHz).

Authors: Zhengnan Li, Claude (Opus 4.6)
Email  : uwa-channels@ofdm.link
License: MIT

Revision history:
  - Feb. 27, 2026: Initial release.
  - Mar.  9, 2026: Unified noise dict. Synthetic beta modeled
                    on real experimental data.
  - Oct. 15, 2026: Exact mixing checks: transpose, time direction,
                    array_index mapping, SaS scale, resampling.
"""

import pytest
import numpy as np
import scipy.signal as sg
from scipy.special import gamma
from uwa_channels import noisegen
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FS = 48000
N = 500000  # For spectral / correlation / distribution tests
N_SHORT = 100000  # For size, resampling, subset, bandpass tests


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


# ====================================================================
#  Synthetic noise struct builder
# ====================================================================


def make_noise_struct(alpha, perturb=0.05):
    """
    Construct a synthetic noise dict modeled on real experimental
    data from a 12-element ULA.

    The beta mixing matrix is built as:
      1. Bandpass FIR (K=65 taps) at fc=13 kHz, R~4.9 kHz.
      2. Scale each (i,j) pair by 0.5^|i-j| (spatial decay).
      3. Add random perturbation scaled by `perturb`.

    Parameters
    ----------
    alpha : float
        Stability index (2 = Gaussian, <2 = impulsive).
    perturb : float, optional
        Perturbation level for beta (default 0.05).
        Set to 0 for exact theoretical correlation tests.
    """
    M = 12
    K = 65
    Fs = 39062.5
    fc = 13000.0
    R = 4882.8125

    # Bandpass FIR base profile
    f_lo = (fc - R / 2) / (Fs / 2)
    f_hi = (fc + R / 2) / (Fs / 2)
    base_profile = sg.firwin(K, [f_lo, f_hi], pass_zero=False)

    # Build beta with spatial decay + perturbation
    rng_state = np.random.get_state()
    np.random.seed(2024)

    decay_rate = 0.5
    beta = np.zeros((M, M, K))
    for i in range(M):
        for j in range(M):
            d = abs(i - j)
            scale = decay_rate**d
            perturbation = perturb * scale * np.random.randn(K)
            beta[i, j, :] = scale * base_profile + perturbation

    np.random.set_state(rng_state)

    # Normalize each channel to unit pseudo-power.
    # Power is the alpha-stable pseudo-power 2*c^2 (c = scale parameter),
    # reducing to the variance when alpha = 2.  Channel i is SaS with
    # c_i^alpha proportional to sum_{j,k} |beta[i,j,k]|^alpha, so scaling each
    # channel's taps to unit alpha-norm yields unit pseudo-power and the
    # pseudo-powers sum to M.  For alpha = 2 this is the L2 (energy) norm.
    for i in range(M):
        beta[i] /= np.sum(np.abs(beta[i]) ** alpha) ** (1.0 / alpha)

    # Per-channel RMS power (mild variation, ratio max/min ~ 1.3)
    rms_power = 3.2e-4 * (1 + 0.1 * np.linspace(-1, 1, M)).reshape(-1, 1)

    return {
        "Fs": np.array([[Fs]]),
        "R": np.array([[R]]),
        "alpha": np.array([[alpha]]),
        "beta": beta,
        "fc": np.array([[fc]]),
        "rms_power": rms_power,
        "version": 1.0,
    }


# ====================================================================
#  Option 1: Pink Gaussian noise
# ====================================================================


@pytest.mark.parametrize("shape", [(100000, 1), (100000, 4), (50000, 8)])
def test_option1_size(shape):
    w = noisegen(shape, FS)
    assert w.shape == shape
    assert np.all(np.isfinite(w))


def test_option1_spectral_slope():
    """Verify ~17 dB/decade slope."""
    w = noisegen((N, 1), FS)
    f, pxx = sg.welch(w[:, 0], FS, nperseg=8192)

    mask = (f >= 100) & (f <= 10000)
    p = np.polyfit(np.log10(f[mask]), 10 * np.log10(pxx[mask]), 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.log10(f[mask]), 10 * np.log10(pxx[mask]), label="Estimated")
    axes[0].plot(
        np.log10(f[mask]),
        np.polyval(p, np.log10(f[mask])),
        "r--",
        label=f"Fit: {p[0]:.1f} dB/dec",
    )
    psd_true = -17 * np.log10(f[mask] / 1e3)
    psd_true = psd_true - np.mean(psd_true) + np.mean(10 * np.log10(pxx[mask]))
    axes[0].plot(np.log10(f[mask]), psd_true, "k--", label="True (-17 dB/dec)")
    axes[0].set_xlabel(r"$\log_{10}(f)$")
    axes[0].set_ylabel("PSD [dB]")
    axes[0].set_title("Pink noise PSD")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].hist(w[:, 0], bins=100, density=True)
    axes[1].set_xlabel("Amplitude")
    axes[1].set_ylabel("PDF")
    axes[1].set_title("Amplitude distribution")

    fig.tight_layout()
    plt.savefig("test_noise_option1_psd.png", dpi=150)
    plt.close(fig)

    assert (
        abs(p[0] - (-17)) < 17 * 0.15
    ), f"Pink noise slope {p[0]:.1f} dB/decade, expected ~-17"


def test_option1_spatial_independence():
    """Channels should be independent.

    Pink noise has most of its power at the lowest frequencies, so it holds
    few independent samples and its sample correlation spreads widely: over
    40 seeds the largest off-diagonal entry reached 0.16 for independent
    channels.  First differences of independent channels are still
    independent but nearly white, so their correlation spreads as 1/sqrt(N):
    over the same seeds the largest entry was 0.004, while a 5% leak of one
    channel into another gives 0.047.
    """
    w = np.diff(noisegen((N, 4), FS), axis=0)
    C = np.corrcoef(w.T)
    off_diag = C - np.eye(4)
    assert np.max(np.abs(off_diag)) < 0.02, "Channels are not independent"


# ====================================================================
#  Option 2: Gaussian noise via beta mixing (alpha = 2)
# ====================================================================


def test_option2_size_and_finite():
    noise = make_noise_struct(2)
    M = noise["beta"].shape[0]
    w = noisegen((N_SHORT, M), FS, list(range(M)), noise)
    assert w.shape == (N_SHORT, M)
    assert np.all(np.isfinite(w))


def test_option2_spatial_correlation():
    """Verify sample correlation matches R = sum_k B_k @ B_k.T."""
    # Use perturb=0 so bandpass preserves the correlation structure
    noise = make_noise_struct(2, perturb=0)
    M = noise["beta"].shape[0]
    K = noise["beta"].shape[2]
    beta = noise["beta"]

    w = noisegen((N, M), FS, list(range(M)), noise)

    # Theoretical correlation from mixing equation
    R_theory = np.zeros((M, M))
    for k in range(K):
        Bk = beta[:, :, k]
        R_theory += Bk @ Bk.T
    d = np.sqrt(np.diag(R_theory))
    C_theory = R_theory / np.outer(d, d)

    C_sample = np.corrcoef(w.T)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    for m in range(min(M, 6)):
        f, pxx = sg.welch(w[:, m], FS, nperseg=8192)
        ax.plot(f / 1e3, 10 * np.log10(pxx), label=f"Ch {m}")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title("Estimated PSD")
    ax.legend(fontsize=7)
    ax.grid(True)

    im = axes[0, 1].imshow(C_theory, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].set_title("Theoretical C")

    im = axes[1, 0].imshow(C_sample, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title("Sample C")

    im = axes[1, 1].imshow(C_sample - C_theory, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].set_title("Error")

    err = np.max(np.abs(C_sample - C_theory))
    fig.suptitle(f"Gaussian mixing: max |error| = {err:.4f}")
    fig.tight_layout()
    plt.savefig("test_noise_option2_correlation.png", dpi=150)
    plt.close(fig)

    assert err < 0.05, f"Correlation mismatch: max |error| = {err:.4f}"


def test_option2_identity_beta_independence():
    """Diagonal-only beta should yield independent channels."""
    M = 12
    noise = make_noise_struct(2)
    noise["beta"] = np.repeat(np.eye(M)[:, :, np.newaxis], 65, axis=2)

    w = noisegen((N_SHORT, M), FS, list(range(M)), noise)
    C = np.corrcoef(w.T)
    off_diag = C - np.eye(M)
    assert (
        np.max(np.abs(off_diag)) < 0.1
    ), "Diagonal-only beta should produce independent channels"


def test_option2_resampling():
    """Correct output length when fs != noise.Fs."""
    noise = make_noise_struct(2)
    # noise.Fs is already 39062.5 != 48000
    M = noise["beta"].shape[0]
    w = noisegen((N_SHORT, M), FS, list(range(M)), noise)
    assert w.shape == (N_SHORT, M)


def test_option2_array_index_subset():
    """Using a subset of array indices."""
    noise = make_noise_struct(2)
    array_index = [1, 5, 9]
    w = noisegen((100000, len(array_index)), FS, array_index, noise)
    assert w.shape == (100000, 3)
    assert np.all(np.isfinite(w))


def test_option2_rms_scaling():
    """Scaling is controlled by the (alpha-normalized) beta coefficients,
    so the generated noise carries unit power per channel and the total
    variance summed over channels equals the number of channels.
    """
    noise = make_noise_struct(2)
    M = noise["beta"].shape[0]
    w = noisegen((N, M), FS, list(range(M)), noise)

    total_var = np.sum(np.var(w, axis=0))  # expect ~M for unit power per channel

    assert abs(total_var - M) / M < 0.15, f"sum(var) = {total_var:.2f}, expected ~{M}"


def test_option2_bandpass():
    """Verify bandpass filtering."""
    noise = make_noise_struct(2)
    fc = float(noise["fc"][0, 0])
    R = float(noise["R"][0, 0])

    w = noisegen((N_SHORT, 2), FS, [0, 1], noise)
    f, pxx = sg.welch(w[:, 0], FS, nperseg=8192)
    pxx_dB = 10 * np.log10(pxx)

    in_band = (f >= fc - R / 2) & (f <= fc + R / 2)
    out_low = (f >= 100) & (f <= fc - R)
    out_high = (f >= fc + R) & (f <= FS / 2 - 100)

    psd_in = np.mean(pxx_dB[in_band])
    psd_out = np.mean(pxx_dB[out_low | out_high])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f / 1e3, pxx_dB)
    ax.axvline((fc - R / 2) / 1e3, color="r", ls="--")
    ax.axvline((fc + R / 2) / 1e3, color="r", ls="--")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(f"Gaussian bandpass: fc={fc/1e3:.0f} kHz, R={R/1e3:.1f} kHz")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig("test_noise_option2_bandpass.png", dpi=150)
    plt.close(fig)

    rejection = psd_in - psd_out
    assert rejection > 18, f"Bandpass rejection {rejection:.1f} dB, expected > 18 dB"


def test_option2_gaussianity():
    """alpha=2 should produce Gaussian output (kurtosis ~ 3)."""
    noise = make_noise_struct(2)
    w = noisegen((N, 1), FS, [0], noise)

    k = _kurtosis(w[:, 0])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(w[:, 0], bins=200, density=True, alpha=0.7)
    x = np.linspace(w.min(), w.max(), 500)
    ax.plot(x, _normpdf(x, 0, np.std(w)), "r", lw=1.5)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("PDF")
    ax.set_title(f"Gaussianity check (kurtosis = {k:.2f})")
    fig.tight_layout()
    plt.savefig("test_noise_option2_gaussianity.png", dpi=150)
    plt.close(fig)

    assert abs(k - 3) / 3 < 0.15, f"Kurtosis {k:.2f}, expected ~3 for Gaussian"


# ====================================================================
#  Option 3: Impulsive (alpha-stable) noise (alpha < 2)
# ====================================================================


def test_option3_size_and_finite():
    noise = make_noise_struct(1.7)
    M = noise["beta"].shape[0]
    w = noisegen((N_SHORT, M), FS, list(range(M)), noise)
    assert w.shape == (N_SHORT, M)
    assert np.all(np.isfinite(w))


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.7, 1.9])
def test_option3_various_alpha(alpha):
    noise = make_noise_struct(alpha)
    w = noisegen((100000, 2), FS, [0, 1], noise)
    assert w.shape == (100000, 2)
    assert np.all(np.isfinite(w)), f"Non-finite values for alpha = {alpha}"


def test_option3_heavier_tail():
    """Lower alpha should produce heavier tails (higher kurtosis)."""
    np.random.seed(1994)
    noise_heavy = make_noise_struct(1.2)
    w_heavy = noisegen((500000, 1), FS, [0], noise_heavy)

    np.random.seed(1994)
    noise_light = make_noise_struct(1.9)
    w_light = noisegen((500000, 1), FS, [0], noise_light)

    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.linspace(-10, 10, 200)
    ax.hist(w_light[:, 0], bins=edges, density=True, alpha=0.7, label=r"$\alpha$=1.9")
    ax.hist(w_heavy[:, 0], bins=edges, density=True, alpha=0.7, label=r"$\alpha$=1.2")
    ax.set_yscale("log")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("PDF")
    ax.set_title("Tail comparison")
    ax.legend()
    fig.tight_layout()
    plt.savefig("test_noise_option3_tails.png", dpi=150)
    plt.close(fig)

    k_heavy = _kurtosis(w_heavy[:, 0])
    k_light = _kurtosis(w_light[:, 0])
    assert (
        k_heavy > k_light
    ), f"alpha=1.2 kurtosis ({k_heavy:.1f}) should exceed alpha=1.9 ({k_light:.1f})"


def test_option3_rms_scaling():
    """For impulsive (alpha < 2) noise the variance is undefined, so power is
    quantified by the alpha-stable pseudo-power 2*c^2, where c is the scale
    parameter; it reduces to the variance when alpha = 2 [Chitre 2007].  beta
    is normalized so each channel carries unit pseudo-power, hence the
    pseudo-powers summed over channels equal the channel count.
    """
    alpha = 1.9
    noise = make_noise_struct(alpha)
    M = noise["beta"].shape[0]
    w = noisegen((N, M), FS, list(range(M)), noise)

    total_pp = sum(_pseudo_power(w[:, m], alpha) for m in range(M))

    assert (
        abs(total_pp - M) / M < 0.15
    ), f"sum(pseudo-power) = {total_pp:.2f}, expected ~{M}"


def test_option3_resampling():
    """Correct output when noise.Fs != fs."""
    noise = make_noise_struct(1.7)
    w = noisegen((100000, 2), FS, [0, 1], noise)
    assert w.shape == (100000, 2)
    assert np.all(np.isfinite(w))


def test_option3_bandpass():
    """Verify bandpass filtering for impulsive noise."""
    noise = make_noise_struct(1.7)
    fc = float(noise["fc"][0, 0])
    R = float(noise["R"][0, 0])

    w = noisegen((N_SHORT, 2), FS, [0, 1], noise)
    f, pxx = sg.welch(w[:, 0], FS, nperseg=8192)
    pxx_dB = 10 * np.log10(pxx)

    in_band = (f >= fc - R / 2) & (f <= fc + R / 2)
    out_low = (f >= 100) & (f <= fc - R)
    out_high = (f >= fc + R) & (f <= FS / 2 - 100)

    psd_in = np.mean(pxx_dB[in_band])
    psd_out = np.mean(pxx_dB[out_low | out_high])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f / 1e3, pxx_dB)
    ax.axvline((fc - R / 2) / 1e3, color="r", ls="--")
    ax.axvline((fc + R / 2) / 1e3, color="r", ls="--")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(f"Impulsive bandpass: fc={fc/1e3:.0f} kHz, R={R/1e3:.1f} kHz")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig("test_noise_option3_bandpass.png", dpi=150)
    plt.close(fig)

    rejection = psd_in - psd_out
    assert rejection > 18, f"Bandpass rejection {rejection:.1f} dB, expected > 18 dB"


def test_option3_alpha2_matches_gaussian():
    """alpha=2 should look Gaussian, alpha=1.5 heavier tails."""
    np.random.seed(42)
    noise_g = make_noise_struct(2)
    w_g = noisegen((500000, 1), FS, [0], noise_g)

    np.random.seed(42)
    noise_i = make_noise_struct(1.5)
    w_i = noisegen((500000, 1), FS, [0], noise_i)

    k_g = _kurtosis(w_g[:, 0])
    k_i = _kurtosis(w_i[:, 0])

    assert (
        k_i > k_g
    ), f"alpha=1.5 kurtosis ({k_i:.1f}) should exceed alpha=2 ({k_g:.1f})"


def test_option3_spatial_correlation():
    """Off-diagonal beta with impulsive noise — structural check.

    For alpha < 2, Pearson correlation overestimates dependence because
    heavy-tailed outliers inflate the sample covariance.  Spearman
    (rank) correlation is robust to this and should track the
    Gaussian-predicted structure more closely.
    """
    from scipy.stats import spearmanr

    noise = make_noise_struct(1.7, perturb=0)
    M = noise["beta"].shape[0]
    K = noise["beta"].shape[2]
    beta = noise["beta"]

    w = noisegen((N, M), FS, list(range(M)), noise)
    C_pearson = np.corrcoef(w.T)
    C_spearman, _ = spearmanr(w)

    # Gaussian-predicted correlation (reference)
    R_theory = np.zeros((M, M))
    for k in range(K):
        Bk = beta[:, :, k]
        R_theory += Bk @ Bk.T
    d = np.sqrt(np.diag(R_theory))
    C_theory = R_theory / np.outer(d, d)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im = axes[0, 0].imshow(C_theory, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[0, 0].set_title("Gaussian-predicted C")

    im = axes[0, 1].imshow(C_pearson, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].set_title(f"Pearson C (α={float(noise['alpha'][0, 0])})")

    im = axes[1, 0].imshow(C_spearman, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title(f"Spearman C (α={float(noise['alpha'][0, 0])})")

    # Correlation decay comparison
    offsets = np.arange(1, M)
    c_theory_d = [np.mean(np.diag(C_theory, d)) for d in offsets]
    c_pearson_d = [np.mean(np.diag(C_pearson, d)) for d in offsets]
    c_spearman_d = [np.mean(np.diag(C_spearman, d)) for d in offsets]
    axes[1, 1].plot(offsets, c_theory_d, "ko-", label="Gaussian theory")
    axes[1, 1].plot(offsets, c_pearson_d, "rs-", label="Pearson")
    axes[1, 1].plot(offsets, c_spearman_d, "b^-", label="Spearman")
    axes[1, 1].set_xlabel("|i-j|")
    axes[1, 1].set_ylabel("Mean correlation")
    axes[1, 1].set_title("Correlation decay")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle(f"Impulsive (α={float(noise['alpha'][0, 0])}) spatial correlation")
    fig.tight_layout()
    plt.savefig("test_noise_option3_correlation.png", dpi=150)
    plt.close(fig)

    # Structural checks (Spearman is the robust metric)
    assert abs(C_spearman[0, 1]) > abs(
        C_spearman[0, -1]
    ), "Adjacent channels should be more correlated than distant ones"
    c_vs_d = [np.mean(np.abs(np.diag(C_spearman, d))) for d in range(1, M)]
    assert c_vs_d[0] > c_vs_d[-1], "Correlation should decay with element separation"


# ====================================================================
#  Mixing structure: exact checks with small hand-built beta
# ====================================================================
#
# The tests above use make_noise_struct, whose beta is nearly symmetric in
# (i, j) and is compared through normalized correlation.  Those tests cannot
# see a transposed beta, a reversed time direction, a wrong array_index
# mapping, or the scale of the alpha-stable drivers.  Each test below is
# built so that one of these errors moves the result outside its tolerance.


def make_beta(M, K):
    """Mixing coefficients that are asymmetric in (i, j) and vary with lag k,
    so that beta and its transpose (in i, j or in k) give different statistics."""
    i, j, k = np.meshgrid(
        np.arange(1, M + 1), np.arange(1, M + 1), np.arange(1, K + 1), indexing="ij"
    )
    return np.cos(1.3 * i + 0.7 * j + 0.5 * k) / k


def make_mixing_noise(beta, Fs=FS, alpha=2.0):
    """Minimal noise dict around a given beta."""
    return {
        "Fs": np.array([[Fs]]),
        "alpha": np.array([[alpha]]),
        "beta": beta,
        "version": 1.0,
    }


def _mix(n, beta, array_index=None, fs=FS, Fs=FS, alpha=2.0, seed=0):
    if array_index is None:
        array_index = list(range(beta.shape[0]))
    np.random.seed(seed)
    noise = make_mixing_noise(beta, Fs, alpha)
    return noisegen((n, len(array_index)), fs, array_index, noise)


def test_mixing_covariance_is_not_transposed():
    """Zero-lag covariance equals sum_k B_k B_k^T, not sum_k B_k^T B_k.

    The raw covariance is compared, not the correlation coefficient, so the
    scale of beta is checked too.
    """
    beta = make_beta(3, 4)
    w = _mix(200000, beta, seed=11)
    C = np.cov(w.T)
    R0 = sum(beta[:, :, k] @ beta[:, :, k].T for k in range(4))
    R0_transposed = sum(beta[:, :, k].T @ beta[:, :, k] for k in range(4))

    err = np.linalg.norm(C - R0) / np.linalg.norm(R0)
    err_transposed = np.linalg.norm(C - R0_transposed) / np.linalg.norm(R0)
    assert err < 0.05, f"Covariance mismatch: relative error {err:.4f}"
    assert err_transposed > 0.5, "Test beta cannot distinguish a transpose"


def test_mixing_runs_forward_in_time():
    """Lag-1 cross-covariance pins the direction of time.

    The generator mixes forward, w_i(n) = sum_{j,k} beta_ij(k) eta_j(n + k),
    the convention under which NoiseModels.jl estimates beta.  Then
    E[w(n) w(n+1)^T] = sum_k B_{k+1} B_k^T.  Mixing backward, eta_j(n - k),
    gives the transpose.  The zero-lag covariance is the same either way.
    """
    beta = make_beta(3, 4)
    w = _mix(200000, beta, seed=11)
    R1 = w[:-1].T @ w[1:] / (len(w) - 1)
    forward = sum(beta[:, :, k + 1] @ beta[:, :, k].T for k in range(3))
    backward = forward.T

    err = np.linalg.norm(R1 - forward) / np.linalg.norm(forward)
    err_backward = np.linalg.norm(R1 - backward) / np.linalg.norm(forward)
    assert err < 0.05, f"Lag-1 covariance mismatch: relative error {err:.4f}"
    assert err_backward > 0.5, "Test beta cannot distinguish the time direction"


def test_mixing_subset_equals_slice():
    """With the same seed, a subset of array_index is the matching columns of
    the full draw, in the order requested."""
    beta = make_beta(4, 3)
    w_all = _mix(500, beta, seed=3)
    w_sub = _mix(500, beta, array_index=[2, 0], seed=3)
    np.testing.assert_allclose(w_sub, w_all[:, [2, 0]])


def test_mixing_array_index_selects_element():
    """Elements with powers 1, 1e2 and 1e4 show which one array_index picked."""
    beta = np.zeros((3, 3, 1))
    for i in range(3):
        beta[i, i, 0] = 10.0**i
    w = _mix(50000, beta, array_index=[2, 0], seed=5)
    p = np.mean(w**2, axis=0)
    assert 0.5e4 < p[0] < 2e4, f"array_index 2 gave power {p[0]:.3g}, expected 1e4"
    assert 0.5 < p[1] < 2, f"array_index 0 gave power {p[1]:.3g}, expected 1"


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.9])
def test_mixing_sas_characteristic_function(alpha):
    """Identity mixing gives SaS noise with scale 1/sqrt(2), whose
    characteristic function is exp(-|t / sqrt(2)|^alpha).

    Unlike kurtosis, which is infinite for alpha < 2, the empirical
    characteristic function is bounded; its standard error here is < 0.003.
    """
    beta = np.eye(2)[:, :, None]
    w = _mix(100000, beta, alpha=alpha, seed=9)
    for t in (0.5, 1.0, 2.0):
        phi = np.mean(np.cos(t * w))
        expected = np.exp(-abs(t / np.sqrt(2)) ** alpha)
        assert (
            abs(phi - expected) < 0.01
        ), f"alpha={alpha}, t={t}: E[cos(tw)] = {phi:.4f}, expected {expected:.4f}"


def test_mixing_resampling_changes_content():
    """Resampling changes the lag-1 autocorrelation, not only the length.

    Measured at Fs = 48 kHz the lag-1 autocorrelation is 0.47; upsampling to
    96 kHz raises it to 0.83 and downsampling to 24 kHz lowers it to 0.27.
    """
    beta = make_beta(2, 3)

    def r1(x):
        return np.sum(x[:-1] * x[1:]) / np.sum(x**2)

    r = {}
    for fs in (24000.0, 48000.0, 96000.0):
        w = _mix(100000, beta, fs=fs, seed=5)
        assert w.shape == (100000, 2)
        assert np.all(np.isfinite(w))
        r[fs] = r1(w[:, 0])
    assert 0.4 < r[48000.0] < 0.55, f"lag-1 at the measured rate: {r[48000.0]:.3f}"
    assert r[96000.0] > 0.7, f"lag-1 at 96 kHz: {r[96000.0]:.3f}"
    assert r[24000.0] < 0.35, f"lag-1 at 24 kHz: {r[24000.0]:.3f}"


# ====================================================================
#  Helpers
# ====================================================================


def _kurtosis(x):
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    m4 = np.mean((x - mu) ** 4)
    return m4 / m2**2


def _normpdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _pseudo_power(x, alpha):
    """Symmetric alpha-stable pseudo-power 2*c^2 estimated via the fractional
    lower-order moment  E|x|^p = C(p, alpha) * c^p  for 0 < p < alpha
    (Samorodnitsky & Taqqu, 1994).  Returns the variance when alpha = 2.
    """
    x = np.asarray(x).ravel()
    if alpha == 2:
        return np.var(x)
    p = 1.0  # must satisfy p < alpha
    C = (
        2**p
        * gamma((p + 1) / 2)
        * gamma(1 - p / alpha)
        / (np.sqrt(np.pi) * gamma(1 - p / 2))
    )
    c = (np.mean(np.abs(x) ** p) / C) ** (1.0 / p)
    return 2 * c**2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
