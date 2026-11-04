import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import scipy.signal as sg
from fractions import Fraction

from .replay import validate_f_resamp


def unpack(fs, array_index, channel, buffer_left=None, buffer_right=None):
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
        re-inserted via interpolation.  A scalar applies to every
        hydrophone; a vector holds one factor per hydrophone in the file
        and is indexed by array_index.

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
        - "f_resamp" : float or ndarray, optional (one factor, or one per
          hydrophone in the file)
        - "version" : ndarray, at least 1.0

    buffer_left : float, optional
        Left buffer, as a fraction of the number of taps.  By default, or
        for None, just enough zeros to hold the largest delay drift, so a
        channel without drift is not padded.

    buffer_right : float, optional
        Right buffer, as buffer_left.

    Returns:
    --------
    ndarray
        Unpacked channel (delay_samples, num_elements, time_samples).

    Raises:
    -------
    ValueError
        If the channel version is below 1.0, fs is not a positive finite
        scalar, array_index is empty, not integer, repeated or out of range,
        a buffer is negative or not a finite scalar, or f_resamp is not
        positive and finite with one factor or one per hydrophone.

    Revision history:
      - Apr.  1, 2025: Initial release.
      - Feb. 27, 2026: Fixed delay insertion for theta_hat vs phi_hat.
      - Jun. 10, 2026: Removed global normalization; fixed spline extrapolation
                       by clamping out-of-range values to zero; replaced
                       resample with PchipInterpolator for phase and drift to
                       eliminate FIR edge-effect ripples.
      - Oct. 27, 2026: phi_hat takes precedence over theta_hat when a channel
                       carries both, as in replay; their phases used to add.
      - Nov.  3, 2026: f_resamp may be a vector with one factor per
                       hydrophone in the file, indexed by array_index; each
                       hydrophone gets its own phase ramp and delay drift.
                       Added validate_inputs, which checks the channel
                       version, fs, array_index, the buffers and f_resamp
                       before unpacking, as in the MATLAB toolbox.
      - Nov.  4, 2026: The buffers default to the room the delay drift needs,
                       instead of 0.1 each, and None asks for that default.
    """

    array_index = np.atleast_1d(array_index)
    validate_inputs(fs, array_index, channel, buffer_left, buffer_right)

    ## Parameters
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = channel["params"]["fs_time"][0, 0]
    fc = channel["params"]["fc"][0, 0]
    h_hat = np.array(channel["h_hat"]["real"] + 1j * channel["h_hat"]["imag"])[
        :, array_index, :
    ]
    T, M, K = h_hat.shape

    N_phi = int(np.ceil(T * fs_delay / fs_time))
    phase_all = np.zeros((N_phi, len(array_index)))
    phase_drift = np.zeros((N_phi, len(array_index)))

    # phi_hat is the whole tracked phase (est.m saves its one trajectory as
    # phi_hat with delay tracking and as theta_hat without), so it takes
    # precedence over theta_hat, as in replay.  theta_hat adds no drift.
    if "phi_hat" in channel:
        phase_all += np.array(channel["phi_hat"])[:, array_index]
        phase_drift += np.array(channel["phi_hat"])[:, array_index]
    elif "theta_hat" in channel:
        phase_all += np.array(channel["theta_hat"])[:, array_index]

    if "f_resamp" in channel:
        # One factor per hydrophone in the file, or one for all of them.
        f_resamp = np.ravel(np.asarray(channel["f_resamp"], dtype=float))
        if f_resamp.size > 1:
            f_resamp = f_resamp[array_index]
        f_resamp_phase = (
            (1 / f_resamp[None, :] - 1)
            * 2
            * np.pi
            * fc
            * np.arange(N_phi)[:, None]
            / fs_delay
        )
        phase_all += f_resamp_phase
        phase_drift += f_resamp_phase

    ## Allocate buffer
    # By default, room for the drift: a tap at delay tau moves to tau - drift, so
    # a positive drift needs room on the left and a negative one on the right.
    drift_samples = phase_drift / (2 * np.pi * fc) * fs_delay
    n_left = int(np.ceil(max(drift_samples.max(), 0)))
    n_right = int(np.ceil(max(-drift_samples.min(), 0)))
    if buffer_left is not None:
        n_left = int(np.ceil(K * buffer_left))
    if buffer_right is not None:
        n_right = int(np.ceil(K * buffer_right))
    h_hat = np.concatenate(
        (np.zeros((T, M, n_left)), h_hat, np.zeros((T, M, n_right))), axis=2
    )
    K = h_hat.shape[2]

    ## Sample rate conversion
    frac_1 = Fraction(fs / fs_time).limit_denominator()
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

    t_orig = np.arange(N_phi) / fs_delay

    for m in range(M):
        h_hat_m = np.squeeze(h_hat[:, m, :])
        h_resampled = sg.resample_poly(
            h_hat_m, frac_1.numerator, frac_1.denominator, axis=0
        )
        t_new = np.arange(h_resampled.shape[0]) / fs

        if has_phase:
            phase_resampled = PchipInterpolator(
                t_orig, phase_all[:, m], extrapolate=True
            )(t_new)
            h_resampled = h_resampled * np.exp(1j * phase_resampled)[:, None]

        if has_drift:
            drift_resampled = PchipInterpolator(
                t_orig, phase_drift[:, m], extrapolate=True
            )(t_new)
            drift = drift_resampled / (2 * np.pi * fc)
            for t in range(h_resampled.shape[0]):
                cs_re = CubicSpline(
                    delays, np.real(h_resampled[t, :]), extrapolate=False
                )
                h_re = np.nan_to_num(cs_re(delays + drift[t]), nan=0.0)
                cs_im = CubicSpline(
                    delays, np.imag(h_resampled[t, :]), extrapolate=False
                )
                h_im = np.nan_to_num(cs_im(delays + drift[t]), nan=0.0)
                h_resampled[t, :] = h_re + 1j * h_im

        unpacked_channel[:, m, :] = h_resampled.T

    return unpacked_channel


def validate_inputs(fs, array_index, channel, buffer_left, buffer_right):
    """Raise ValueError if an input to unpack is invalid."""
    version = np.asarray(channel["version"], dtype=float).ravel()[0]
    if not version >= 1.0:
        raise ValueError(
            "The minimum version of the channel matrix is 1.0, "
            f"and you have {version:.1f}."
        )

    if not _is_real_scalar(fs) or not fs > 0:
        raise ValueError("fs must be a positive, finite scalar.")

    N = np.asarray(channel["h_hat"]["real"]).shape[1]
    index = np.asarray(array_index)
    if (
        index.size == 0
        or not np.issubdtype(index.dtype, np.number)
        or not np.all(index == np.round(index))
    ):
        raise ValueError("array_index must be integers.")
    if len(np.unique(index)) != index.size:
        raise ValueError("array_index must not contain duplicate indices.")
    if index.min() < 0 or index.max() >= N:
        raise ValueError(f"array_index must be valid channel indices in [0, {N}).")

    for buffer in (buffer_left, buffer_right):
        if buffer is not None and (not _is_real_scalar(buffer) or not buffer >= 0):
            raise ValueError("The buffers must be non-negative, finite scalars.")

    if "f_resamp" in channel:
        validate_f_resamp(channel["f_resamp"], N)


def _is_real_scalar(x):
    """True for a finite real number, or an array holding exactly one."""
    x = np.asarray(x)
    return (
        x.ndim == 0
        and np.issubdtype(x.dtype, np.number)
        and np.isrealobj(x)
        and bool(np.isfinite(x))
    )


# [EOF]
