import numpy as np
import scipy.signal as sg
from fractions import Fraction


def generate_noise(input_shape, fs, noise=None, array_index=[0]):
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
        w /= np.sqrt(pwr(w))

    # Generate noise according to statistics collected during experiment.
    elif noise is not None:
        frac = Fraction(fs / Fs).limit_denominator()
        signal_size = np.array(input_shape)
        signal_size[0] = np.ceil(signal_size[0] / fs * Fs).astype(int)
        h = np.array(noise["h"])
        h /= np.sqrt(pwr(h.T))[:, None]
        n = np.random.randn(signal_size[0], noise["sigma"].shape[0]) @ np.linalg.cholesky(noise["sigma"])
        w = np.zeros(signal_size)
        for m in range(signal_size[1]):
            w[:, m] = sg.fftconvolve(n[:, array_index[m]], h[array_index[m], :], "same")
        w = sg.resample_poly(w, frac.numerator, frac.denominator)

    else:
        raise ValueError("Wrong noise_option.")

    w /= np.sqrt(np.sum(pwr(w)))
    w = w[: input_shape[0], :]
    return w


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


# [EOF]
