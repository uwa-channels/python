import numpy as np
import scipy.signal as sg
from fractions import Fraction
from scipy.stats import levy_stable  # noqa: F401  (kept for impulsive path)


def noisegen(input_shape, fs, array_index=(0,), noise=None):
    """
    Generate underwater acoustic noise.

    This function produces noise signals based on the provided parameters.

    1. **Pink noise** (default, noise=None): Independent Gaussian noise
       with a 17 dB/decade roll-off across array elements.
    2. **Mixing-coefficient noise** (noise dict provided): Spatially-
       correlated noise shaped by mixing coefficients beta.  The
       stability index alpha determines the distribution:
         alpha == 2  => Gaussian (via Box-Muller / randn).
         alpha <  2  => Symmetric alpha-stable (impulsive).

    Parameters
    ----------
    input_shape : tuple
        Shape of the output noise array (time_samples, num_channels).
    fs : float
        Sampling frequency of the output signal in Hz.
    array_index : list of int, optional
        Indices of the array elements.  Default is (0,).
    noise : dict, optional
        Dictionary with the unified noise struct fields:
          Fs        - Sampling rate at which statistics were measured [Hz].
          R         - Signal bandwidth [Hz].
          alpha     - Stability index (2 = Gaussian, <2 = impulsive).
          beta      - Mixing coefficients, shape (M, M, K).  Normalized so
                      each channel carries unit pseudo-power (2*c^2, the
                      alpha-stable scale measure that reduces to the variance
                      when alpha = 2); the pseudo-powers summed over channels
                      equal the channel count M.  beta also encodes the
                      bandpass spectral shaping.
          fc        - Center frequency [Hz].
          version   - Noise struct version (>= 1.0).

    Returns
    -------
    ndarray
        Noise array of shape `input_shape`.

    Raises
    ------
    ValueError
        If an invalid noise configuration is provided.

    Examples
    --------
    For detailed examples, refer to the scripts in the `examples` folder.

    Revision history
    ----------------
      - Apr.  1, 2025: Initial release.
      - Feb. 27, 2026: Fixed spectrum mirror and np.astype usage.
      - Mar.  9, 2026: Unified noise dict.  Removed Cholesky path.
    """

    if noise is None:
        w = _noise_pink(input_shape, fs)
    else:
        array_index = list(array_index)
        _validate_inputs(input_shape, noise, array_index)

        Fs = float(noise["Fs"][0, 0])
        w = _noise_mixing(input_shape, fs, Fs, noise, array_index)

    return w


def _validate_inputs(input_shape, noise, array_index):
    version = float(np.asarray(noise["version"]).ravel()[0])
    if version < 1.0:
        raise ValueError(
            f"The minimum version of the noise dict is 1.0, "
            f"and you have {version}."
        )
    if len(set(array_index)) != len(array_index):
        raise ValueError("array_index must contain unique entries.")
    if len(input_shape) < 2 or input_shape[1] != len(array_index):
        raise ValueError(
            f"input_shape[1] ({input_shape[1]}) must equal "
            f"len(array_index) ({len(array_index)})."
        )
    M = np.asarray(noise["beta"]).shape[0]
    if max(array_index) >= M:
        raise ValueError(
            f"max(array_index) ({max(array_index)}) must be less than "
            f"M ({M})."
        )


def _noise_pink(input_shape, fs):
    """Independent pink Gaussian noise (-17 dB per decade), unit power per channel,
    so the summed power over channels equals the channel count."""
    nfft = 4096
    fmin = 0
    fmax = fs / 2
    f = np.linspace(fs / 2 / nfft, fs / 2, nfft)
    H_dB = -17 * np.log10(f / 1e3)
    H_oneside = 10 ** (H_dB / 10)
    H_oneside[: int(np.floor(fmin / (fs / 2 / nfft)))] = 0
    H_oneside[int(np.ceil(fmax / (fs / 2 / nfft))) :] = 0
    H = np.sqrt(np.concatenate((H_oneside, H_oneside[:0:-1])))
    h = np.real(np.fft.fftshift(np.fft.ifft(H)))
    h /= np.sqrt(np.sum(h**2))
    w = np.random.randn(input_shape[0], input_shape[1])
    for m in range(input_shape[1]):
        w[:, m] = sg.fftconvolve(w[:, m], h, "same")
    return w


def _noise_mixing(input_shape, fs, Fs, noise, array_index):
    """Mixing-coefficient noise (Gaussian or impulsive).

    Time-domain mixing:
        w[n, i] = sum_j sum_k beta[i, j, k] * z[n+k, j]
    Implemented as one BLAS matmul per tap k, restricted to the requested
    output rows.  Memory is O(K * M) for z and O(K * len(array_index)) for w;
    no large FFT arrays are allocated.

    The driver z has unit pseudo-power: for alpha = 2 it is standard Gaussian
    (Var[z] = 1); for alpha < 2 it is SaS with scale 1/sqrt(2) (matching
    MATLAB stabrnd's c = 1/sqrt(2)).  Per-channel scaling is carried entirely
    by the (normalized) mixing coefficients beta.
    """
    alpha = float(noise["alpha"][0, 0])
    beta = np.asarray(noise["beta"])

    # Ensure shape is (M, M, K): handle a MATLAB-style (K, M, M) layout.
    if beta.ndim == 3 and beta.shape[0] != beta.shape[1]:
        beta = np.transpose(beta, (1, 2, 0))

    frac = Fraction(fs / Fs).limit_denominator()
    signal_size = list(input_shape)
    signal_size[0] = int(np.ceil(signal_size[0] / fs * Fs))

    K = signal_size[0]
    M = beta.shape[0]
    K_mix = beta.shape[2]

    if alpha == 2:
        z = np.random.randn(K + K_mix, M)
    else:
        z = levy_stable.rvs(alpha, 0, scale=1.0 / np.sqrt(2), size=(K + K_mix, M))

    beta_sub = beta[array_index, :, :]   # (Nout, M, K_mix)

    w = np.zeros((K, len(array_index)))
    for k in range(K_mix):
        w += z[k:k + K, :] @ beta_sub[:, :, k].T

    w = sg.resample_poly(w, frac.numerator, frac.denominator, axis=0)
    w = w[: input_shape[0], :]
    return w


# [EOF]
