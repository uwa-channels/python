import numpy as np
import scipy.signal as sg
from fractions import Fraction


def generate_noise(input, fs, array_index, noise, noise_option):

    Fs = noise["Fs"][0, 0]  # Sampling rate of the original recording.
    fc_exp = noise["fc"][0, 0]  # Center frequency of the original recording.
    R_exp = noise["R"][0, 0]  # Symbol rate of the original recording.
    fmin = fc_exp - R_exp / 2 - 100
    fmax = fc_exp + R_exp / 2 + 100

    frac = Fraction(fs / Fs).limit_denominator()
    signal_size = np.array(input.shape)
    signal_size[0] = np.astype(np.ceil(signal_size[0] / fs * Fs), int)

    # Generate textbook style noise: independent Gaussian noise (17dB/decade) across array elements.
    if noise_option == 1:
        nfft = 4096
        f = np.linspace(Fs / 2 / nfft, Fs / 2, nfft)
        H_dB = -17 * np.log10(f / 1e3)
        H_oneside = 10 ** (H_dB / 10)
        H_oneside[: np.astype(np.floor(fmin / (Fs / 2 / nfft)), int)] = 0
        H_oneside[np.astype(np.ceil(fmax / (Fs / 2 / nfft)), int) :] = 0
        H = np.sqrt(np.concatenate((H_oneside, H_oneside[1::-1])))
        h = np.fft.fftshift(np.fft.irfft(H))
        w = np.random.randn(signal_size[0], signal_size[1])
        for m in range(signal_size[1]):
            w[:, m] = np.convolve(w[:, m], h, "same")
    # Generate noise according to statistics collected during experiment.
    elif noise_option == 2:
        n = np.random.randn(signal_size[0], len(noise["sigma"])) @ np.linalg.cholesky(noise["sigma"])
        w = np.zeros(signal_size)
        for m in range(signal_size[1]):
            w[:, m] = np.convolve(n[:, array_index[m]], noise["h"][array_index[m], :], "same")
    # Generate noise according to the fitted power spectral density of the measured noise.
    elif noise_option == 3:
        n = np.random.randn(signal_size[0], len(noise["sigma"])) @ np.linalg.cholesky(noise["sigma"])
        w = np.zeros(signal_size)
        for m in range(signal_size[1]):
            w[:, m] = np.convolve(n[:, array_index[m]], noise["h_fit"][array_index[m], :], "same")

    w = sg.resample_poly(w, frac.numerator, frac.denominator)
    w /= np.sqrt(np.sum(pwr(w)))
    w = w[: input.shape[0], :]
    return w


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)
