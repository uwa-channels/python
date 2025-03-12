import numpy as np
from scipy.interpolate import CubicSpline
import scipy.signal as sg
from fractions import Fraction


def replay(input, fs, array_index, channel, start=None):
    # Unpacking variables
    h_hat_real = np.array(channel["h_hat"]["real"])[:, array_index, :]
    h_hat_imag = np.array(channel["h_hat"]["imag"])[:, array_index, :]
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = channel["params"]["fs_time"][0, 0]
    fc = channel["params"]["fc"][0, 0]
    M = len(array_index)
    L = h_hat_real.shape[2]
    if "theta_hat" in channel.keys():
        theta_hat = np.array(channel["theta_hat"])[:, array_index]
    else:
        resampling_factor = channel["resampling_factor"][0, 0]

    # Convert baseband and resample the signal to fs_delay
    frac = Fraction(fs_delay / fs).limit_denominator()
    baseband = input * np.exp(-2j * np.pi * fc * np.arange(input.shape[0]) / fs)
    baseband = sg.resample_poly(baseband, frac.numerator, frac.denominator)
    T = baseband.shape[0]

    # Assign random start point in time
    T_max = h_hat_real.shape[0] / fs_time * fs_delay * 1.0
    if start is None:
        start = np.random.randint(low=0, high=T_max - T - L)
    print(f"Start = {start}")

    # Convolution
    buffer = np.zeros((L - 1,))
    baseband = np.concatenate((buffer, baseband, buffer))
    output = np.zeros((T + L, M), dtype=complex)
    signal_time = np.arange(start, start + T + L) / fs_delay
    signal_start = np.floor(np.min(signal_time) * fs_time).astype(int)
    signal_end = np.ceil(np.max(signal_time) * fs_time).astype(int)
    frac1 = Fraction(fs_delay / fs_time).limit_denominator()
    for m in range(M):
        ir_real = sg.resample_poly(
            h_hat_real[signal_start:signal_end, m, ::-1], frac1.numerator, frac1.denominator, axis=0
        )
        ir_imag = sg.resample_poly(
            h_hat_imag[signal_start:signal_end, m, ::-1], frac1.numerator, frac1.denominator, axis=0
        )
        ir = ir_real + 1j * ir_imag

        if "theta_hat" in channel.keys():
            for t in np.arange(T + L - 1):
                output[t, m] = (ir[t, :] @ baseband[t : t + L]) * np.exp(1j * theta_hat[t, m])
            drift = theta_hat[np.arange(start, start + T + L), m] / (2 * np.pi * fc)
            output[:, m] = CubicSpline(signal_time, output[:, m])(signal_time + drift)
        else:
            for t in np.arange(T + L - 1):
                output[t, m] = np.sum(ir[t, :] * baseband[t : t + L])

    # Resample to match the original sampling rate and upshift to fc
    output = sg.resample_poly(output, frac.denominator, frac.numerator)
    output = np.real(output * np.exp(2j * np.pi * fc * np.arange(len(output))[:, None] / fs))

    # Resample in passband if needed
    if "theta_hat" not in channel.keys():
        frac_resample = Fraction(resampling_factor).limit_denominator()
        output = sg.resample_poly(output, frac_resample.numerator, frac_resample.denominator)

    output /= np.sqrt(np.sum(pwr(output)))

    return output


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


# [EOF]
