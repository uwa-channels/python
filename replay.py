import numpy as np
from scipy.interpolate import CubicSpline
import scipy.signal as sg
from fractions import Fraction


def replay(input, fs, array_index, channel, start=None):
    ## Step 1: unpacking variables
    h_hat_real = np.array(channel["h_hat"]["real"])
    h_hat_imag = np.array(channel["h_hat"]["imag"])
    theta_hat = np.array(channel["theta_hat"])
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = channel["params"]["fs_time"][0, 0]
    fc = channel["params"]["fc"][0, 0]
    M = len(array_index)
    Ts = 1.0 / fs_delay
    Tr = Ts / 2.0
    offset = 200

    T_max, _, L = h_hat_real.shape
    T_max = np.floor((T_max - 1) / fs_time * fs_delay)

    ## Step 2: convert baseband and resample the signal to fs_delay
    baseband = input * np.exp(-2j * np.pi * fc * np.arange(input.shape[0]) / fs)
    frac = Fraction(fs_delay / fs).limit_denominator()
    baseband = sg.resample_poly(baseband, frac.numerator, frac.denominator)
    T = baseband.shape[0]
    baseband = np.concatenate(
        (
            np.zeros(
                L - 1,
            ),
            baseband,
            np.zeros(
                L - 1,
            ),
        )
    )

    ## Step 3: pre-allocation
    output = np.zeros((baseband.shape[0] + offset, M), dtype=complex)

    ## Step 4: assign random start point in time
    if start is None:
        start = np.random.randint(low=0, high=T_max - T)

    ## Step 5: convolution
    actual_time = np.arange(h_hat_real.shape[0]) / fs_time
    for m in range(M):
        ir_real = CubicSpline(actual_time, np.squeeze(h_hat_real[:, m, :]))(np.arange(start, start + T) / fs_delay)
        ir_imag = CubicSpline(actual_time, np.squeeze(h_hat_imag[:, m, :]))(np.arange(start, start + T) / fs_delay)
        ir = ir_real + 1j * ir_imag

        for t in np.arange(T):
            output[t + offset, m] = np.sum(ir[t, :] * baseband[t : t + L]) * np.exp(
                1j * theta_hat[t + start, array_index[m]]
            )

    ## Step 6: insert the delay back
    y = np.zeros(output.shape, dtype=complex)
    frac = Fraction(Ts / Tr).limit_denominator()
    output = sg.resample_poly(output, frac.numerator, frac.denominator)
    t = np.arange(T)
    for m in range(M):
        tn = t * Ts + theta_hat[t + start, array_index[m]] / (2 * np.pi * fc)
        tln = np.floor(tn / Tr) * Tr
        trn = np.ceil(tn / Tr) * Tr
        alpha = (tn - tln) / Tr
        y[t + offset, m] = (1 - alpha) * output[np.astype(np.floor(tln / Tr), int) + offset, m] + alpha * output[
            np.astype(np.ceil(trn / Tr), int) + offset, m
        ]
    output = y

    frac = Fraction(fs / fs_delay).limit_denominator()
    output = sg.resample_poly(output, frac.numerator, frac.denominator)
    output = np.real(output * np.exp(2j * np.pi * fc * np.arange(len(output))[:, None] / fs))
    output /= np.sqrt(np.sum(pwr(output)))

    return output


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


# [EOF]
