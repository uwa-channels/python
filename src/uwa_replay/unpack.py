import numpy as np
from scipy.interpolate import CubicSpline
import scipy.signal as sg
from fractions import Fraction


def unpack(fs, array_index, channel, buffer_left=0.1, buffer_right=0.1):
    """
    Unpack an underwater acoustic channel.

    Parameters:
    -----------
    fs : float
        Target sampling frequency along time axis in Hz.

    array_index : list of int
        Indices of the array elements to process.

    channel : dict
        Dictionary containing channel characteristics. Expected keys in `channel`:

        - "h_hat" : dict
            - "real" : ndarray
                Real part of the channel impulse response.
            - "imag" : ndarray
                Imaginary part of the channel impulse response.
        - "params" : dict
            - "fs_delay" : float
                Sampling frequency of the delay domain.
            - "fs_time" : float
                Sampling frequency in the time domain.
            - "fc" : float
                Carrier frequency.
        - "theta_hat" : ndarray
            Phase estimates for phase correction.
        - "f_resamp" : float, optional
            Factor for additional passband resampling.

    buffer_left : float, optional
        Left buffer fraction for impulse response padding. Default is 0.1.

    buffer_right : float, optional
        Right buffer fraction for impulse response padding. Default is 0.1.

    Returns:
    --------
    ndarray
        Unpacked and resampled channel with shape (delay_samples, num_elements, time_samples).

    Examples:
    ---------
    For more detailed examples, refer to the corresponding scripts in the `examples` folder.

    """

    ## Parameters
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = channel["params"]["fs_time"][0, 0]
    fc = channel["params"]["fc"][0, 0]
    h_hat = np.array(channel["h_hat"]["real"] + 1j * channel["h_hat"]["imag"])[:, array_index, :]
    T, M, K = h_hat.shape

    theta_hat = np.zeros((np.ceil(T * fs_delay / fs_time).astype(int), len(array_index)))
    if "theta_hat" in channel:
        theta_hat += np.array(channel["theta_hat"])[:, array_index]
    if "f_resamp" in channel:
        theta_hat += (
            (1 / channel["f_resamp"][0, 0] - 1)
            * 2
            * np.pi
            * fc
            * np.arange(np.ceil(T * fs_delay / fs_time))[:, None]
            / fs_delay
        )

    ## Allocate some buffer
    h_hat = np.concatenate(
        (
            np.zeros((T, M, (np.ceil(K * buffer_left).astype(int)))),
            h_hat,
            np.zeros((T, M, (np.ceil(K * buffer_right).astype(int)))),
        ),
        axis=2,
    )
    K = h_hat.shape[2]

    ## Sample rate conversion
    frac_1 = Fraction(fs / fs_time).limit_denominator()
    frac_2 = Fraction(fs / fs_delay).limit_denominator()
    delays = np.arange(K) / fs_delay

    ## Unpack the channel for every element
    unpacked_channel = np.zeros(
        (K, len(array_index), np.ceil(T * frac_1.numerator / frac_1.denominator).astype(int)), dtype=complex
    )
    for m in range(M):
        h_hat_m = np.squeeze(h_hat[:, m, :])
        if "theta_hat" in channel or "f_resamp" in channel:
            theta_hat_resampled = sg.resample_poly(theta_hat[:, m], frac_2.numerator, frac_2.denominator, axis=0)
            drift = theta_hat_resampled / (2 * np.pi * fc)
            unpacked = (
                sg.resample_poly(h_hat_m, frac_1.numerator, frac_1.denominator, axis=0)
                * np.exp(1j * theta_hat_resampled)[:, None]
            )
            for t in range(unpacked.shape[0]):
                unpacked_re = CubicSpline(delays, np.real(unpacked[t, :]))(delays + drift[t])
                unpacked_im = CubicSpline(delays, np.imag(unpacked[t, :]))(delays + drift[t])
                unpacked[t, :] = unpacked_re + 1j * unpacked_im
        else:
            unpacked = sg.resample_poly(h_hat_m, frac_1.numerator, frac_1.denominator, axis=0)
        unpacked_channel[:, m, :] = unpacked.T
    unpacked_channel /= np.max(np.abs(unpacked))

    return unpacked_channel


# [EOF]
