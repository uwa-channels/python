import pytest
import numpy as np
import scipy.signal as sg
from uwa_replay import noisegen
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 48000
N = 1000000


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1994)


def make_gaussian_noise(M, Fs):
    return {
        "Fs": np.array([[Fs]]),
        "sigma": np.eye(M),
        "h": np.random.randn(M, 100),
        "version": 1.0,
    }


def make_impulsive_noise(alpha, M, Fs):
    return {
        "Fs": np.array([[Fs]]),
        "alpha": alpha,
        "beta": np.repeat(np.eye(M)[None, :, :], 65, axis=0),
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
    axes[0].plot(np.log10(f[mask]), np.polyval(p, np.log10(f[mask])), "r--", label=f"Fit: {p[0]:.1f} dB/dec")
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
#  Option 2: Colored spatially-correlated Gaussian noise
# ====================================================================


def test_option2_size_and_finite():
    noise = make_gaussian_noise(4, FS)
    w = noisegen((N, 4), FS, list(range(4)), noise)
    assert w.shape == (N, 4)
    assert np.all(np.isfinite(w))


def test_option2_spatial_correlation():
    """Verify correlation structure and spectral shape."""
    M = 4
    A = np.random.randn(M, M)
    sigma = A @ A.T + M * np.eye(M)

    h_single = np.random.randn(1, 100)
    noise = {
        "Fs": np.array([[FS]]),
        "sigma": sigma,
        "h": np.tile(h_single, (M, 1)),  # same filter, all channels
        "version": 1.0,
    }

    w = noisegen((N, M), FS, list(range(M)), noise)
    C_sample = np.corrcoef(w.T)
    d = np.sqrt(np.diag(sigma))
    C_truth = sigma / np.outer(d, d)

    err = np.max(np.abs(C_sample - C_truth))

    # Plot: 2x2 — estimated PSD, true PSD, sample corr, truth corr
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Estimated PSD
    ax = axes[0, 0]
    h_filt = noise["h"][0, :]
    w_true_f, H_true = sg.freqz(h_filt, 1, worN=4096, fs=FS)
    pxx_all = []
    for m in range(M):
        f, pxx = sg.welch(w[:, m], FS, nperseg=8192)
        ax.plot(f / 1e3, 10 * np.log10(pxx), label=f"Ch {m}")
        pxx_all.append(pxx)
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title("Estimated PSD")
    ax.legend(fontsize=7)
    ax.grid(True)

    # True PSD
    ax = axes[0, 1]
    psd_true_0 = np.abs(H_true) ** 2 * sigma[0, 0]
    scale = np.median(pxx_all[0]) / np.median(psd_true_0)
    for m in range(M):
        psd_true = np.abs(H_true) ** 2 * sigma[m, m] * scale
        ax.plot(w_true_f / 1e3, 10 * np.log10(psd_true), label=f"Ch {m}")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title("True PSD")
    ax.legend(fontsize=7)
    ax.grid(True)

    # Sample correlation
    ax = axes[1, 0]
    im = ax.imshow(C_sample, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Sample correlation")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")

    # Truth correlation
    ax = axes[1, 1]
    im = ax.imshow(C_truth, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Truth correlation")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")

    fig.suptitle(f"Option 2: max corr error = {err:.3f}")
    fig.tight_layout()
    plt.savefig("fig_noise_option2_correlation.png", dpi=150)
    plt.close(fig)

    assert err < 0.05, f"Correlation mismatch: max error = {err:.3f}"


def test_option2_resampling():
    """Correct output length when fs != noise.Fs."""
    noise = make_gaussian_noise(2, 96000)
    w = noisegen((N, 2), FS, [0, 1], noise)
    assert w.shape == (N, 2)


def test_option2_array_index_subset():
    """Using a subset of array indices."""
    noise = make_gaussian_noise(6, FS)
    w = noisegen((100000, 3), FS, [1, 3, 5], noise)
    assert w.shape == (100000, 3)
    assert np.all(np.isfinite(w))


# ====================================================================
#  Option 3: Impulsive (alpha-stable) noise
# ====================================================================


def test_option3_size_and_finite():
    noise = make_impulsive_noise(1.7, 3, FS)
    w = noisegen((N, 3), FS, list(range(3)), noise)
    assert w.shape == (N, 3)
    assert np.all(np.isfinite(w))


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.7, 1.9])
def test_option3_various_alpha(alpha):
    noise = make_impulsive_noise(alpha, 2, FS)
    w = noisegen((100000, 2), FS, [0, 1], noise)
    assert w.shape == (100000, 2)
    assert np.all(np.isfinite(w)), f"Non-finite values for alpha = {alpha}"


def test_option3_heavier_tail():
    """Lower alpha should produce heavier tails (higher kurtosis)."""
    np.random.seed(1994)
    noise_heavy = make_impulsive_noise(1.2, 1, FS)
    w_heavy = noisegen((500000, 1), FS, [0], noise_heavy)

    np.random.seed(1994)
    noise_light = make_impulsive_noise(1.9, 1, FS)
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
    assert k_heavy > k_light, f"alpha=1.2 kurtosis ({k_heavy:.1f}) should exceed alpha=1.9 ({k_light:.1f})"


def test_option3_rms_scaling():
    """Verify rms_power scaling."""
    noise = make_impulsive_noise(1.9, 2, FS)  # closer to Gaussian for stable RMS
    noise["rms_power"] = np.array([[2], [0.5]])

    w = noisegen((500000, 2), FS, [0, 1], noise)
    rms_ratio = np.sqrt(np.mean(w[:, 0] ** 2)) / np.sqrt(np.mean(w[:, 1] ** 2))

    assert abs(rms_ratio - 4) / 4 < 0.5, f"RMS ratio {rms_ratio:.2f}, expected ~4"


def test_option3_resampling():
    """Correct output when noise.Fs != fs."""
    noise = make_impulsive_noise(1.7, 2, 96000)
    w = noisegen((100000, 2), FS, [0, 1], noise)
    assert w.shape == (100000, 2)
    assert np.all(np.isfinite(w))


# ====================================================================
#  Helpers
# ====================================================================


def _kurtosis(x):
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    m4 = np.mean((x - mu) ** 4)
    return m4 / m2**2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
