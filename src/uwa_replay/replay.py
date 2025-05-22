import numpy as np
from scipy.interpolate import CubicSpline
import scipy.signal as sg
from fractions import Fraction


def replay(input, fs, array_index, channel, start=None):
    """
    Simulate the replay of a passband signal through an underwater acoustic channel.

    Parameters:
    -----------
    input : ndarray
        Real passband input signal.

    fs : float
        Sampling frequency of the input signal in Hz.

    array_index : list of int
        Indices of the array elements to simulate.

    channel : dict
        Dictionary containing channel characteristics. Expected keys in `channel`:

        - "h_hat" : dict
            - "real" : ndarray
                Real part of the estimated channel impulse response.
            - "imag" : ndarray
                Imaginary part of the estimated channel impulse response.
        - "params" : dict
            - "fs_delay" : float
                Sampling frequency of the delay domain.
            - "fs_time" : float
                Sampling frequency in the time domain.
            - "fc" : float
                Carrier frequency.
        - "theta_hat" : ndarray, optional
            Phase estimates for phase correction.
        - "f_resamp" : float, optional
            Factor for additional passband resampling.

    start : int, optional
        Random starting point for signal propagation. If None, a random point is chosen.

    Returns:
    --------
    ndarray
        Simulated replay output with dimensions (samples, array_elements).

    Raises:
    -------
    ValueError
        If channel parameters are missing or inconsistent.

    Examples:
    ---------
    For more detailed examples, refer to the corresponding scripts in the `examples` folder.

    """

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

    # Convert to baseband and resample the signal to fs_delay
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
    channel_time = np.arange(h_hat_real.shape[0]) / fs_time 
    signal_time = np.arange(start, start + T + L) / fs_delay
    for m in range(M):
        ir_real = CubicSpline(channel_time, np.squeeze(h_hat_real[:, m, ::-1]))(signal_time)
        ir_imag = CubicSpline(channel_time, np.squeeze(h_hat_imag[:, m, ::-1]))(signal_time)
        ir = ir_real + 1j * ir_imag
        if "theta_hat" in channel.keys():
            for t in np.arange(T + L - 1):
                output[t, m] = (ir[t, :] @ baseband[t : t + L]) * np.exp(1j * theta_hat[t, m])
            drift = theta_hat[np.arange(start, start + T + L), m] / (2 * np.pi * fc)
            output[:, m] = CubicSpline(signal_time, output[:, m])(signal_time + drift)
        else:
            for t in np.arange(T + L - 1):
                output[t, m] = ir[t, :] @ baseband[t : t + L]

    # Resample to match the original sampling rate and upshift to fc
    output = sg.resample_poly(output, frac.denominator, frac.numerator)
    output = np.real(output * np.exp(2j * np.pi * fc * np.arange(len(output))[:, None] / fs))

    # Resample in passband if needed
    if "f_resamp" in channel.keys():
        frac_resample = Fraction(channel["f_resamp"][0, 0]).limit_denominator()
        output = sg.resample_poly(output, frac_resample.numerator, frac_resample.denominator)

    output /= np.sqrt(np.sum(pwr(output)))

    return output


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


# [EOF]
