import numpy as np
import scipy.signal as sg
from fractions import Fraction
from scipy.stats import levy_stable


def generate_noise(input_shape, fs, array_index=[0], noise=None):
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
        w -= np.mean(w, axis=0)
        w /= np.sqrt(pwr(w))

    # Generate noise according to statistics collected during experiment.
    elif noise is not None and "sigma" in noise.keys():
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
        w -= np.mean(w, axis=0)
        w /= np.sqrt(np.sum(pwr(w)))
        w = w[: input_shape[0], :]

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
        w /= np.sqrt(np.sum(pwr(w)))
        w = w[: input_shape[0], :]

    else:
        raise ValueError("Wrong noise_option.")

    return w


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


# [EOF]
