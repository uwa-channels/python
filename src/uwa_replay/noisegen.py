import numpy as np
import scipy.signal as sg
from fractions import Fraction
from scipy.stats import levy_stable


def noisegen(input_shape, fs, array_index=[0], noise=None):
    """
    Generate various types of noise signals for array processing.

    This function produces noise signals based on the provided parameters. It can generate three types of noise:

    1. **Textbook-style noise** (default behavior): Independent Gaussian noise with a 17 dB/decade roll-off across array elements.
    2. **Experiment-based noise**: Noise shaped by statistical data collected during an experiment, including custom covariance.
    3. **Impulsive noise**: Noise generated using a symmetric alpha-stable distribution, allowing for heavy-tailed behavior.

    Parameters:
    -----------
    input_shape : tuple
        Shape of the output noise array (time_samples, num_channels).

    fs : float
        Sampling frequency of the output signal in Hz.

    array_index : list of int, optional
        Indices of the array elements to which noise should be applied. Default is [0].

    noise : dict, optional
        Dictionary specifying noise characteristics. If None, textbook style pink Gaussian noise is generated.

        Expected keys in `noise`:

        - "Fs" : ndarray
            Sampling frequency of the input noise template.
        - "sigma" : ndarray, optional
            Covariance matrix for experiment-based noise.
        - "h" : ndarray, optional
            Impulse responses for experiment-based noise.
        - "alpha" : float, optional
            Stability parameter for alpha-stable impulsive noise (0 < alpha <= 2).
        - "beta" : ndarray, optional
            Coefficient array for shaping impulsive noise.

    Returns:
    --------
    ndarray
        Noise array of shape `input_shape` with the specified noise characteristics.

    Raises:
    -------
    ValueError
        If an invalid noise configuration is provided.

    Examples:
    ---------
    For more detailed examples, refer to the corresponding scripts in the `examples` folder.

    """

    if noise is not None:
        Fs = noise["Fs"][0, 0]

    # Generate textbook style noise: independent Gaussian noise (17dB/decade) across array elements.
    if noise is None:
        white_noise = np.random.randn(input_shape[0], input_shape[1])
        freqs = np.fft.rfftfreq(input_shape[0], d=1 / fs)
        spectrum = np.fft.rfft(white_noise, axis=0)
        alpha = 1.7  # 1/f^alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            filter_shape = np.where(freqs > 0, 1 / (freqs ** (alpha / 2)), 1)
        filtered_spectrum = spectrum * filter_shape[:, None]
        w = np.fft.irfft(filtered_spectrum, n=input_shape[0], axis=0)

    # Generate noise according to statistics collected during experiment.
    elif noise is not None and "sigma" in noise.keys():
        frac = Fraction(fs / Fs).limit_denominator()
        signal_size = np.array(input_shape)
        signal_size[0] = np.ceil(signal_size[0] / fs * Fs).astype(int)
        h = np.array(noise["h"])
        n = np.random.randn(signal_size[0], noise["sigma"].shape[0]) @ np.linalg.cholesky(noise["sigma"])
        w = np.zeros(signal_size)
        for m in range(signal_size[1]):
            w[:, m] = sg.fftconvolve(n[:, array_index[m]], h[array_index[m], :], "same")
        w = sg.resample_poly(w, frac.numerator, frac.denominator)
        w = w[: input_shape[0], :]

    # Generate impulsive noise.
    elif noise is not None and "alpha" in noise.keys():
        alpha = noise["alpha"]
        beta = np.array(noise["beta"]).T
        frac = Fraction(fs / Fs).limit_denominator()
        signal_size = np.array(input_shape)
        signal_size[0] = np.astype(np.ceil(signal_size[0] / fs * Fs), int)
        K = signal_size[0]
        N = beta.shape[0]

        z = levy_stable.rvs(alpha, 0, size=(K + beta.shape[2], N))
        w = np.zeros((K, N))
        for i in range(N):
            for j in range(N):
                for k in range(beta.shape[2]):
                    w[:, i] = w[:, i] + beta[i, j, k] * z[k : k + K, j]

        w = w[:, array_index]
        w = sg.resample_poly(w, frac.numerator, frac.denominator)
        w = w[: input_shape[0], :]

    else:
        raise ValueError("Wrong noise_option.")

    return w


# [EOF]
