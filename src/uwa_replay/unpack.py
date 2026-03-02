import numpy as np
from scipy.interpolate import CubicSpline
import scipy.signal as sg
from fractions import Fraction


def unpack(fs, array_index, channel, buffer_left=0.1, buffer_right=0.1):
    """
    Unpack an underwater acoustic channel.

    Handles three tracking modes:
      theta_hat (phase tracking only):
        h_hat already contains drifting taps. Only phase is re-inserted.
        No delay interpolation is performed.
      phi_hat (delay tracking):
        h_hat is static. Both phase and delay drift are re-inserted.
      f_resamp:
        Adds a linear phase ramp. Its delay contribution is always
        re-inserted via interpolation.

    Parameters:
    -----------
    fs : float
        Target sampling frequency along time axis in Hz.

    array_index : list of int
        Indices of the array elements to process.

    channel : dict
        Dictionary containing channel characteristics. Expected keys:

        - "h_hat" : dict with "real" and "imag" ndarrays
        - "params" : dict with "fs_delay", "fs_time", "fc"
        - "theta_hat" : ndarray, optional (phase only)
        - "phi_hat" : ndarray, optional (phase + delay)
        - "f_resamp" : float, optional

    buffer_left : float, optional
        Left buffer fraction. Default is 0.1.

    buffer_right : float, optional
        Right buffer fraction. Default is 0.1.

    Returns:
    --------
    ndarray
        Unpacked channel (delay_samples, num_elements, time_samples).

    Revision history:
      - Apr.  1, 2025: Initial release.
      - Feb. 27, 2026: Fixed delay insertion for theta_hat vs phi_hat.
    """

    ## Parameters
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = channel["params"]["fs_time"][0, 0]
    fc = channel["params"]["fc"][0, 0]
    h_hat = np.array(channel["h_hat"]["real"] + 1j * channel["h_hat"]["imag"])[:, array_index, :]
    T, M, K = h_hat.shape

    N_phi = int(np.ceil(T * fs_delay / fs_time))
    phase_all = np.zeros((N_phi, len(array_index)))
    phase_drift = np.zeros((N_phi, len(array_index)))

    if "theta_hat" in channel:
        phase_all += np.array(channel["theta_hat"])[:, array_index]
        # theta_hat does NOT contribute to phase_drift

    if "phi_hat" in channel:
        phase_all += np.array(channel["phi_hat"])[:, array_index]
        phase_drift += np.array(channel["phi_hat"])[:, array_index]

    if "f_resamp" in channel:
        f_resamp_phase = (1 / channel["f_resamp"][0, 0] - 1) * 2 * np.pi * fc * np.arange(N_phi)[:, None] / fs_delay
        phase_all += f_resamp_phase
        phase_drift += f_resamp_phase

    ## Allocate buffer
    h_hat = np.concatenate(
        (
            np.zeros((T, M, int(np.ceil(K * buffer_left)))),
            h_hat,
            np.zeros((T, M, int(np.ceil(K * buffer_right)))),
        ),
        axis=2,
    )
    K = h_hat.shape[2]

    ## Sample rate conversion
    frac_1 = Fraction(fs / fs_time).limit_denominator()
    frac_2 = Fraction(fs / fs_delay).limit_denominator()
    delays = np.arange(K) / fs_delay

    has_phase = "theta_hat" in channel or "phi_hat" in channel or "f_resamp" in channel
    has_drift = "phi_hat" in channel or "f_resamp" in channel

    ## Unpack
    unpacked_channel = np.zeros(
        (
            K,
            len(array_index),
            int(np.ceil(T * frac_1.numerator / frac_1.denominator)),
        ),
        dtype=complex,
    )

    for m in range(M):
        h_hat_m = np.squeeze(h_hat[:, m, :])
        h_resampled = sg.resample_poly(h_hat_m, frac_1.numerator, frac_1.denominator, axis=0)

        if has_phase:
            phase_resampled = sg.resample_poly(phase_all[:, m], frac_2.numerator, frac_2.denominator, axis=0)
            h_resampled = h_resampled * np.exp(1j * phase_resampled)[:, None]

        if has_drift:
            drift_resampled = sg.resample_poly(phase_drift[:, m], frac_2.numerator, frac_2.denominator, axis=0)
            drift = drift_resampled / (2 * np.pi * fc)
            for t in range(h_resampled.shape[0]):
                h_re = CubicSpline(delays, np.real(h_resampled[t, :]))(delays + drift[t])
                h_im = CubicSpline(delays, np.imag(h_resampled[t, :]))(delays + drift[t])
                h_resampled[t, :] = h_re + 1j * h_im

        unpacked_channel[:, m, :] = h_resampled.T

    unpacked_channel /= np.max(np.abs(unpacked_channel))

    return unpacked_channel


# [EOF]
