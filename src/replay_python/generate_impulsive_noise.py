import numpy as np
import scipy.signal as sg
from scipy.stats import levy_stable
from fractions import Fraction


def generate_impulsive_noise(input, fs, array_index, noise):
    Fs = noise["Fs"][0, 0]
    alpha = noise["alpha"]
    beta = np.array(noise["beta"]).T
    print(beta.shape)

    frac = Fraction(fs / Fs).limit_denominator()
    signal_size = np.array(input.shape)
    signal_size[0] = np.astype(np.ceil(signal_size[0] / fs * Fs), int)

    K = signal_size[0]
    N = beta.shape[0]

    z = levy_stable.rvs(alpha, 0, size=(K+beta.shape[2], N))
    w = np.zeros((K, N))

    for i in range(N):
        for j in range(N):
            for k in range(beta.shape[2]):
                w[:, i] = w[:, i] + beta[i, j, k] * z[k:k+K, j]

    w = w[:, array_index]
    w = sg.resample_poly(w, frac.numerator, frac.denominator)
    w /= np.sqrt(np.sum(pwr(w)))
    w = w[: input.shape[0], :]
    return w


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


# [EOF]
