import numpy as np
import scipy.signal as sg
from fractions import Fraction
from scipy.stats import levy_stable


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
          beta      - Mixing coefficients, shape (M, M, K).
          fc        - Center frequency [Hz].
          rms_power - Per-channel RMS power scaling, shape (M, 1).
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
    elif noise is not None:
        Fs = float(noise["Fs"])
        w = _noise_mixing(input_shape, fs, Fs, noise, array_index)

        # Per-channel RMS power scaling
        rms_power = np.asarray(noise["rms_power"]).ravel()
        w = w * rms_power[list(array_index)]

        # Bandpass filtering (zero-phase to match MATLAB's bandpass)
        fc = float(noise["fc"])
        R = float(noise["R"])
        fl = fc - R / 2 * 1.1
        fh = fc + R / 2 * 1.1
        sos = sg.butter(21, [fl, fh], btype="band", fs=fs, output="sos")
        w = sg.sosfiltfilt(sos, w, axis=0)
    else:
        raise ValueError("Wrong noise option.")

    return w


def _noise_pink(input_shape, fs):
    """Independent pink Gaussian noise (17 dB/decade)."""
    nfft = 4096
    fmin = 0
    fmax = fs / 2

    f = np.linspace(fs / 2 / nfft, fs / 2, nfft)
    H_dB = -17 * np.log10(f / 1e3)
    H_oneside = 10 ** (H_dB / 10)
    H_oneside[: int(np.floor(fmin / (fs / 2 / nfft)))] = 0
    H_oneside[int(np.ceil(fmax / (fs / 2 / nfft))) :] = 0
    H = np.sqrt(np.concatenate((H_oneside, H_oneside[-2::-1])))
    h = np.real(np.fft.fftshift(np.fft.ifft(H)))
    w = np.random.randn(input_shape[0], input_shape[1])
    for m in range(input_shape[1]):
        w[:, m] = sg.fftconvolve(w[:, m], h, "same")
    return w


def _noise_mixing(input_shape, fs, Fs, noise, array_index):
    """Mixing-coefficient noise (Gaussian or impulsive via stabrnd/levy_stable)."""
    alpha = float(noise["alpha"])
    beta = np.asarray(noise["beta"])

    # Handle MATLAB-style (K, M, M) or Python-style (M, M, K)
    # Ensure shape is (M, M, K)
    if beta.ndim == 3 and beta.shape[0] != beta.shape[1]:
        beta = np.transpose(beta, (1, 2, 0))

    frac = Fraction(fs / Fs).limit_denominator()
    signal_size = list(input_shape)
    signal_size[0] = int(np.ceil(signal_size[0] / fs * Fs))

    K = signal_size[0]
    M = beta.shape[0]
    K_mix = beta.shape[2]

    # Generate iid innovations
    if alpha == 2:
        z = np.random.randn(K + K_mix, M)
    else:
        z = levy_stable.rvs(alpha, 0, size=(K + K_mix, M))

    # Apply mixing: w[n, i] = sum_j sum_k beta[i,j,k] * z[n+k, j]
    # This is a cross-correlation of z[:,j] with beta[i,j,:].
    # In frequency domain: W_i(f) = sum_j Z_j(f) * conj(B_{ij}(f))
    from scipy.fft import next_fast_len, rfft, irfft

    nfft = next_fast_len(K + K_mix)

    # Forward FFT of innovation channels (M transforms, reused for all i)
    Z = rfft(z[: K + K_mix, :], n=nfft, axis=0)  # (nfft//2+1, M)

    # FFT of beta taps (short, cheap)
    B = rfft(beta, n=nfft, axis=2)  # (M, M, nfft//2+1)

    w = np.zeros((K, len(array_index)))
    for idx, i in enumerate(array_index):
        W_i = np.sum(Z * np.conj(B[i, :, :].T), axis=1)  # (nfft//2+1,)
        w[:, idx] = irfft(W_i, n=nfft)[:K]

    w = sg.resample_poly(w, frac.numerator, frac.denominator, axis=0)
    w = w[: input_shape[0], :]
    return w


# [EOF]
