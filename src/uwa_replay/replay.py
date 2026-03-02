import numpy as np
from scipy.interpolate import CubicSpline
import scipy.signal as sg
from fractions import Fraction


def replay(input, fs, array_index, channel, start=None):
    """
    Simulate the replay of a passband signal through an underwater acoustic channel.

    Handles three tracking modes:
      theta_hat (phase tracking only):
        h_hat already contains drifting taps. Only phase is re-inserted
        via exp(+j*theta_hat). No delay interpolation is performed.
      phi_hat (delay tracking):
        h_hat is static. Both phase and delay drift are re-inserted.
        Delay drift: drift = phi_hat / (2*pi*fc).
      No tracking:
        Pure convolution with h_hat. Only valid for static channels.

    Parameters:
    -----------
    input : ndarray
        Real passband input signal.

    fs : float
        Sampling frequency of the input signal in Hz.

    array_index : list of int
        Indices of the array elements to simulate.

    channel : dict
        Dictionary containing channel characteristics. Expected keys:

        - "h_hat" : dict
            - "real" : ndarray
            - "imag" : ndarray
        - "params" : dict
            - "fs_delay", "fs_time", "fc" : float (wrapped in np.array)
        - "theta_hat" : ndarray, optional
            Phase estimates (phase tracking only, no delay compensation).
        - "phi_hat" : ndarray, optional
            Phase estimates (delay tracking: phase + delay compensation).
        - "f_resamp" : float, optional
            Factor for additional passband resampling.

    start : int, optional
        Starting index for signal propagation. If None, a random point is chosen.

    Returns:
    --------
    ndarray
        Simulated replay output with dimensions (samples, array_elements).

    Revision history:
      - Apr.  1, 2025: Initial release.
      - Feb. 27, 2026: Fixed delay insertion for theta_hat vs phi_hat.
    """

    # Validate inputs
    validate_inputs(input, fs, array_index, channel)

    # Unpacking variables
    h_hat_real = np.array(channel["h_hat"]["real"])[:, array_index, :]
    h_hat_imag = np.array(channel["h_hat"]["imag"])[:, array_index, :]
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = channel["params"]["fs_time"][0, 0]
    fc = channel["params"]["fc"][0, 0]
    M = len(array_index)
    L = h_hat_real.shape[2]

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

        if "phi_hat" in channel:
            # Delay tracking: phase + delay re-insertion
            phi_hat = np.array(channel["phi_hat"])[:, array_index[m]]
            for t in np.arange(T + L - 1):
                output[t, m] = (ir[t, :] @ baseband[t : t + L]) * np.exp(1j * phi_hat[t + start])
            drift = phi_hat[np.arange(start, start + T + L)] / (2 * np.pi * fc)
            output[:, m] = CubicSpline(signal_time, output[:, m])(signal_time + drift)
        elif "theta_hat" in channel:
            # Phase tracking only: no delay interpolation
            theta_hat = np.array(channel["theta_hat"])[:, array_index[m]]
            for t in np.arange(T + L - 1):
                output[t, m] = (ir[t, :] @ baseband[t : t + L]) * np.exp(1j * theta_hat[t + start])
        else:
            # No tracking: pure convolution
            for t in np.arange(T + L - 1):
                output[t, m] = ir[t, :] @ baseband[t : t + L]

    # Resample to match the original sampling rate and upshift to fc
    output = sg.resample_poly(output, frac.denominator, frac.numerator)
    output = np.real(output * np.exp(2j * np.pi * fc * np.arange(len(output))[:, None] / fs))

    # Resample in passband if needed
    if "f_resamp" in channel:
        frac_resample = Fraction(channel["f_resamp"][0, 0]).limit_denominator()
        output = sg.resample_poly(output, frac_resample.numerator, frac_resample.denominator)

    output /= np.sqrt(np.sum(pwr(output)))

    return output


def pwr(x):
    return np.mean(np.abs(x) ** 2, axis=0)


def validate_inputs(input, fs, array_index, channel):
    assert (
        channel["version"][0, 0] >= 1.0
    ), f"The minimum version of the channel matrix is 1.0, and you have {channel['version']:.1f}."

    T = input.shape[0]
    T = T / fs
    T_max, N, _ = channel["h_hat"]["real"].shape
    T_max = T_max / channel["params"]["fs_time"][0, 0]

    assert T < T_max, f"Duration of the input signal, {T * 1e3:.2f}ms, should be no larger than {T_max * 1e3:.2f}ms."

    assert len(set(array_index)) == len(array_index), "index contains duplicate values."
    assert max(array_index) <= N, f"array_index must be positive integers and cannot exceed {N}."
    assert input.ndim == 1, "The maximum supported number of channels is 1."


# [EOF]
