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
"""

import pytest
import numpy as np
import scipy.signal as sg
from uwa_replay import noisegen
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 48000
N = 500000       # For spectral / correlation / distribution tests
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

    # Per-channel RMS power (mild variation, ratio max/min ~ 1.3)
    rms_power = 3.2e-4 * (1 + 0.1 * np.linspace(-1, 1, M)).reshape(-1, 1)

    return {
        "Fs": Fs,
        "R": R,
        "alpha": alpha,
        "beta": beta,
        "fc": fc,
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
    plt.savefig("fig_noise_option1_psd.png", dpi=150)
    plt.close(fig)

    assert abs(p[0] - (-17)) < 17 * 0.15, f"Pink noise slope {p[0]:.1f} dB/decade, expected ~-17"


def test_option1_spatial_independence():
    """Channels should be independent."""
    w = noisegen((N, 4), FS)
    C = np.corrcoef(w.T)
    off_diag = C - np.eye(4)
    assert np.max(np.abs(off_diag)) < 0.1, "Channels are not independent"


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
    plt.savefig("fig_noise_option2_correlation.png", dpi=150)
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
    assert np.max(np.abs(off_diag)) < 0.05, "Diagonal-only beta should produce independent channels"


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
    """Verify rms_power scaling with identity beta."""
    M = 12
    noise = make_noise_struct(2)
    noise["beta"] = np.repeat(np.eye(M)[:, :, np.newaxis], 65, axis=2)
    rms = np.ones((M, 1))
    rms[0] = 3
    rms[1] = 1
    noise["rms_power"] = rms

    w = noisegen((500000, 2), FS, [0, 1], noise)
    rms_ratio = np.sqrt(np.mean(w[:, 0] ** 2)) / np.sqrt(np.mean(w[:, 1] ** 2))

    assert abs(rms_ratio - 3) / 3 < 0.15, f"RMS ratio {rms_ratio:.2f}, expected ~3"


def test_option2_bandpass():
    """Verify bandpass filtering."""
    noise = make_noise_struct(2)
    fc = noise["fc"]
    R = noise["R"]

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
    plt.savefig("fig_noise_option2_bandpass.png", dpi=150)
    plt.close(fig)

    rejection = psd_in - psd_out
    assert rejection > 20, f"Bandpass rejection {rejection:.1f} dB, expected > 20 dB"


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
    plt.savefig("fig_noise_option2_gaussianity.png", dpi=150)
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
    plt.savefig("fig_noise_option3_tails.png", dpi=150)
    plt.close(fig)

    k_heavy = _kurtosis(w_heavy[:, 0])
    k_light = _kurtosis(w_light[:, 0])
    assert k_heavy > k_light, (
        f"alpha=1.2 kurtosis ({k_heavy:.1f}) should exceed alpha=1.9 ({k_light:.1f})"
    )


def test_option3_rms_scaling():
    """Verify rms_power scaling with identity beta."""
    M = 12
    noise = make_noise_struct(1.9)
    noise["beta"] = np.repeat(np.eye(M)[:, :, np.newaxis], 65, axis=2)
    rms = np.ones((M, 1))
    rms[0] = 2
    rms[1] = 0.5
    noise["rms_power"] = rms

    w = noisegen((500000, 2), FS, [0, 1], noise)
    rms_ratio = np.sqrt(np.mean(w[:, 0] ** 2)) / np.sqrt(np.mean(w[:, 1] ** 2))

    assert abs(rms_ratio - 4) / 4 < 0.5, f"RMS ratio {rms_ratio:.2f}, expected ~4"


def test_option3_resampling():
    """Correct output when noise.Fs != fs."""
    noise = make_noise_struct(1.7)
    w = noisegen((100000, 2), FS, [0, 1], noise)
    assert w.shape == (100000, 2)
    assert np.all(np.isfinite(w))


def test_option3_bandpass():
    """Verify bandpass filtering for impulsive noise."""
    noise = make_noise_struct(1.7)
    fc = noise["fc"]
    R = noise["R"]

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
    plt.savefig("fig_noise_option3_bandpass.png", dpi=150)
    plt.close(fig)

    rejection = psd_in - psd_out
    assert rejection > 20, f"Bandpass rejection {rejection:.1f} dB, expected > 20 dB"


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

    assert k_i > k_g, f"alpha=1.5 kurtosis ({k_i:.1f}) should exceed alpha=2 ({k_g:.1f})"


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
    axes[0, 1].set_title(f"Pearson C (α={noise['alpha']})")

    im = axes[1, 0].imshow(C_spearman, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title(f"Spearman C (α={noise['alpha']})")

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

    fig.suptitle(f"Impulsive (α={noise['alpha']}) spatial correlation")
    fig.tight_layout()
    plt.savefig("fig_noise_option3_correlation.png", dpi=150)
    plt.close(fig)

    # Structural checks (Spearman is the robust metric)
    assert abs(C_spearman[0, 1]) > abs(C_spearman[0, -1]), (
        "Adjacent channels should be more correlated than distant ones"
    )
    c_vs_d = [np.mean(np.abs(np.diag(C_spearman, d))) for d in range(1, M)]
    assert c_vs_d[0] > c_vs_d[-1], "Correlation should decay with element separation"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
