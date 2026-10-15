import functools
import os
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import scipy.fft as sf
import scipy.signal as sg

# Rows of driver samples drawn from each child generator.  The output for a
# given seed depends on this constant, not on the number of threads.
_CHUNK = 1 << 13


def noisegen(input_shape, fs, array_index=(0,), noise=None):
    """
    Generate underwater acoustic noise.

    This function produces noise signals based on the provided parameters.

    1. **Pink noise** (default, noise=None): Independent Gaussian noise
       with a 17 dB/decade roll-off across array elements.
    2. **Mixing-coefficient noise** (noise dict provided): Spatially-
       correlated noise shaped by mixing coefficients beta.  The
       stability index alpha determines the distribution:
         alpha == 2  => Gaussian.
         alpha <  2  => Symmetric alpha-stable (impulsive).

    Parameters
    ----------
    input_shape : tuple
        Shape of the output noise array (time_samples, num_channels).
    fs : float
        Sampling frequency of the output signal in Hz.
    array_index : list of int, optional
        Indices of the array elements.  Default is (0,).
    noise : dict, optional
        Dictionary with the unified noise struct fields:
          Fs        - Sampling rate at which statistics were measured [Hz].
          R         - Signal bandwidth [Hz].
          alpha     - Stability index (2 = Gaussian, <2 = impulsive).
          beta      - Mixing coefficients, shape (M, M, K).  Normalized so
                      each channel carries unit pseudo-power (2*c^2, the
                      alpha-stable scale measure that reduces to the variance
                      when alpha = 2); the pseudo-powers summed over channels
                      equal the channel count M.  beta also encodes the
                      bandpass spectral shaping.
          fc        - Center frequency [Hz].
          version   - Noise struct version (>= 1.0).

    Returns
    -------
    ndarray
        Noise array of shape `input_shape`.

    Raises
    ------
    ValueError
        If an invalid noise configuration is provided.

    Notes
    -----
    The random drivers are seeded from NumPy's global random state, so
    ``np.random.seed`` makes the output reproducible.  The work is spread
    over all available CPU cores; the output does not depend on how many.

    Examples
    --------
    For detailed examples, refer to the scripts in the `examples` folder.

    Revision history
    ----------------
      - Apr.  1, 2025: Initial release.
      - Feb. 27, 2026: Fixed spectrum mirror and np.astype usage.
      - Mar.  9, 2026: Unified noise dict.  Removed Cholesky path.
      - Jul.  8, 2026: Vectorized pink noise generation; the per-channel
                       convolution loop is replaced by a single FFT
                       convolution across all channels.  Output is unchanged.
      - Oct. 15, 2026: Multithreaded, block-wise generation.  Mixing and pink
                       filtering use overlap-save FFT convolution in place in
                       the driver buffer, resampling runs in aligned blocks,
                       and alpha-stable drivers are drawn with the
                       Chambers-Mallows-Stuck formula instead of
                       scipy.stats.levy_stable.  Same statistics; the samples
                       for a given seed differ from earlier releases.
    """

    pool = _pool()
    if noise is None:
        w = _noise_pink(input_shape, fs, pool)
    else:
        array_index = list(array_index)
        _validate_inputs(input_shape, noise, array_index)

        Fs = float(noise["Fs"][0, 0])
        w = _noise_mixing(input_shape, fs, Fs, noise, array_index, pool)

    return w


def _validate_inputs(input_shape, noise, array_index):
    version = float(np.asarray(noise["version"]).ravel()[0])
    if version < 1.0:
        raise ValueError(
            f"The minimum version of the noise dict is 1.0, " f"and you have {version}."
        )
    if len(set(array_index)) != len(array_index):
        raise ValueError("array_index must contain unique entries.")
    if len(input_shape) < 2 or input_shape[1] != len(array_index):
        raise ValueError(
            f"input_shape[1] ({input_shape[1]}) must equal "
            f"len(array_index) ({len(array_index)})."
        )
    M = np.asarray(noise["beta"]).shape[0]
    if max(array_index) >= M:
        raise ValueError(
            f"max(array_index) ({max(array_index)}) must be less than " f"M ({M})."
        )


def _noise_pink(input_shape, fs, pool):
    """Independent pink Gaussian noise (-17 dB per decade), unit power per channel,
    so the summed power over channels equals the channel count."""
    # Column-wise 'same' convolution, w[n] = sum_m h[m] x[n + c - m] with
    # c = (L - 1) // 2 and x zero outside [0, N).  As a correlation with the
    # reversed filter it is w[n] = sum_k h[L-1-k] z[n + k], where z is x with
    # L-1-c zeros before and c zeros after.
    N, nch = input_shape[0], input_shape[1]
    L = _PINK_TAPS
    c = (L - 1) // 2
    z = np.zeros((N + L - 1, nch))
    _draw_drivers(z[L - 1 - c : L - 1 - c + N], 2, pool)
    nfft = _nfft(L, N)
    _correlate_in_place(z, N, L, _pink_transform(fs, nfft), nfft, pool)
    return z[:N]


_PINK_TAPS = 2 * 4096 - 1


@functools.lru_cache(maxsize=16)
def _pink_transform(fs, nfft_conv):
    """Length-nfft_conv rfft of the pink filter h, laid out (frequency, 1)."""
    nfft = 4096
    fmin = 0
    fmax = fs / 2
    f = np.linspace(fs / 2 / nfft, fs / 2, nfft)
    H_dB = -17 * np.log10(f / 1e3)
    H_oneside = 10 ** (H_dB / 10)
    H_oneside[: int(np.floor(fmin / (fs / 2 / nfft)))] = 0
    H_oneside[int(np.ceil(fmax / (fs / 2 / nfft))) :] = 0
    H = np.sqrt(np.concatenate((H_oneside, H_oneside[:0:-1])))
    h = np.real(np.fft.fftshift(np.fft.ifft(H)))
    h /= np.sqrt(np.sum(h**2))
    G = sf.rfft(h, nfft_conv)[:, None]
    G.flags.writeable = False
    return G


def _noise_mixing(input_shape, fs, Fs, noise, array_index, pool):
    """Mixing-coefficient noise (Gaussian or impulsive).

    Time-domain mixing, forward in time:
        w[n, i] = sum_j sum_k beta[i, j, k] * z[n+k, j]
    computed by overlap-save FFT convolution in blocks, written over the
    driver buffer z as it goes, then resampled from Fs to fs in blocks.

    The driver z has unit pseudo-power: for alpha = 2 it is standard Gaussian
    (Var[z] = 1); for alpha < 2 it is SaS with scale 1/sqrt(2) (matching
    MATLAB stabrnd's c = 1/sqrt(2)).  Per-channel scaling is carried entirely
    by the (normalized) mixing coefficients beta.
    """
    alpha = float(noise["alpha"][0, 0])
    beta = np.asarray(noise["beta"])

    # Ensure shape is (M, M, K).  h5py reports a MATLAB [M, M, K] array with
    # every axis reversed, i.e. as (K, M, M) with A[k, j, i] = beta_ij(k), so
    # the inverse is (2, 1, 0).  It is not (1, 2, 0): that leaves beta
    # transposed in (i, j), so the mixing computes sum_j beta_ji(k) eta_j
    # instead of sum_j beta_ij(k) eta_j.  beta is not symmetric in (i, j) --
    # for blue_noise the two differ by 55% of the peak coefficient -- so the
    # spatial covariance of the generated noise comes out wrong.
    if beta.ndim == 3 and beta.shape[0] != beta.shape[1]:
        beta = np.transpose(beta, (2, 1, 0))

    frac = Fraction(fs / Fs).limit_denominator()
    p, q = frac.numerator, frac.denominator

    N = input_shape[0]
    K = N if p == q == 1 else int(np.ceil(N / fs * Fs))
    M = beta.shape[0]
    K_mix = beta.shape[2]
    Nout = len(array_index)

    z = np.empty((K + K_mix - 1, M))
    _draw_drivers(z, alpha, pool)

    # Correlating with beta is convolving with its taps reversed.  The
    # transform is laid out (frequency, j, i) for one matmul per block.
    nfft = _nfft(K_mix, K)
    taps = np.ascontiguousarray(
        np.asarray(beta, dtype=float)[array_index, :, ::-1].transpose(2, 1, 0)
    )
    G = sf.rfft(taps, nfft, axis=0)
    _correlate_in_place(z, K, K_mix, G, nfft, pool)
    w = z[:K, :Nout]

    if p == q == 1:
        return w if Nout == M else np.ascontiguousarray(w)
    return _resample(w, p, q, N, pool)


_POOL = None
_POOL_PID = None


def _pool():
    """A thread pool kept across calls, since starting the threads costs about
    as much as generating a short record.  A forked child (multiprocessing)
    inherits the pool object but not its threads, so it gets a new one."""
    global _POOL, _POOL_PID
    if _POOL is None or _POOL_PID != os.getpid():
        _POOL = ThreadPoolExecutor(_n_workers(), thread_name_prefix="noisegen")
        _POOL_PID = os.getpid()
    return _POOL


def _n_workers():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _nfft(L, K):
    """FFT length for overlap-save of K outputs with an L-tap filter: at least
    4 L, so that the L-1 discarded samples per block are at most a quarter, but
    otherwise short, since short blocks stay in cache (4096 measured fastest
    for 129 and 513 taps), and no longer than one block covering all K."""
    return sf.next_fast_len(min(max(4 * L, 4096), K + L - 1), real=True)


def _draw_drivers(z, alpha, pool):
    """Fill the C-contiguous array z with i.i.d. drivers: standard Gaussian for
    alpha = 2, else symmetric alpha-stable with scale 1/sqrt(2).

    The rows are drawn in chunks of _CHUNK, each from its own child generator,
    so the chunks can be drawn in parallel.  The children are seeded from
    NumPy's global random state, which np.random.seed controls.
    """
    starts = range(0, z.shape[0], _CHUNK)
    entropy = np.random.randint(0, 2**32, size=4).tolist()
    rngs = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(entropy).spawn(len(starts))
    ]

    def draw(start, rng):
        out = z[start : start + _CHUNK]
        if alpha == 2:
            rng.standard_normal(out=out)
        else:
            _draw_sas(rng, alpha, out)

    list(pool.map(draw, starts, rngs))


def _draw_sas(rng, alpha, out):
    """Symmetric alpha-stable variates with scale 1/sqrt(2), written into out:

        x = sin(alpha u) / cos(u)^(1/alpha) * (cos((1 - alpha) u) / e)^((1 - alpha) / alpha)

    with u uniform on [-pi/2, pi/2) and e standard exponential (Chambers,
    Mallows & Stuck, JASA 1976), as stabrnd in the MATLAB toolbox.  At
    alpha = 1 this is tan(u), the Cauchy distribution.
    """
    u = rng.random(out.shape)
    u -= 0.5
    u *= np.pi
    e = rng.standard_exponential(out.shape)
    np.cos((1 - alpha) * u, out=out)
    out /= e
    out **= (1 - alpha) / alpha
    np.cos(u, out=e)
    e **= 1 / alpha
    out /= e
    np.multiply(u, alpha, out=e)
    np.sin(e, out=e)
    out *= e
    out *= np.sqrt(0.5)


def _correlate_in_place(z, K, L, G, nfft, pool):
    """Overwrite z[:K, :Nout] with w[n, i] = sum_j sum_k g[i, j, k] z[n + k, j].

    G is the length-nfft rfft of the reversed taps g[..., ::-1], laid out
    (frequency, j, i), or (frequency, 1) for the same filter on every column.
    z must have K + L - 1 rows.

    Overlap-save in blocks of B = nfft - L + 1 outputs.  Block b reads rows
    [s, s + B + L - 1) and writes rows [s, s + B).  The only rows it reads that
    another block writes are the first L - 1 of block b + 1, so those are
    copied out before any block runs; the blocks are then independent.
    """
    B = nfft - L + 1
    starts = list(range(0, K, B))
    heads = [z[s : s + L - 1].copy() for s in starts[1:]] + [None]
    diagonal = G.ndim == 2
    inner = max(1, _n_workers() // len(starts))  # FFT threads when blocks are few
    Nout = z.shape[1] if diagonal else G.shape[2]

    def block(s, head):
        b = min(B, K - s)
        seg = np.zeros((nfft, z.shape[1]))
        if head is None:
            seg[: b + L - 1] = z[s : s + b + L - 1]
        else:
            seg[:b] = z[s : s + b]
            seg[b : b + L - 1] = head
        Z = sf.rfft(seg, axis=0, workers=inner)
        W = Z * G if diagonal else (Z[:, None, :] @ G)[:, 0, :]
        z[s : s + b, :Nout] = sf.irfft(W, nfft, axis=0)[L - 1 : L - 1 + b]

    list(pool.map(block, starts, heads))


@functools.lru_cache(maxsize=8)
def _resample_filter(p, q):
    """The filter resample_poly designs for p/q by default, designed once.
    Passed back as window=, resample_poly applies the same h *= p."""
    max_rate = max(p, q)
    h = sg.firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    h.flags.writeable = False
    return h


def _resample(w, p, q, N, pool):
    """sg.resample_poly(w, p, q, axis=0)[:N], computed in blocks.

    resample_poly's output m depends on inputs within 10 * max(p, q) / p
    samples of m * q / p.  An input segment starting at a multiple of q maps
    output m' of the segment onto output m' + start * p / q of the whole, so
    segments padded by that support and aligned to q give the same samples as
    one call on the whole signal.
    """
    K, Nout = w.shape
    h = _resample_filter(p, q)
    margin = q * int(np.ceil((10 * max(p, q) / p + 2) / q))
    Bin = q * max(1, int(np.ceil((1 << 13) / q)))
    out = np.empty((N, Nout))

    def block(n0):
        m0 = n0 * p // q
        m1 = min((n0 + Bin) * p // q, N)
        if m1 <= m0:
            return
        a = max(0, n0 - margin)
        seg = np.ascontiguousarray(w[a : min(K, n0 + Bin + margin)].T)
        y = sg.resample_poly(seg, p, q, axis=1, window=h)
        off = a * p // q
        out[m0:m1] = y[:, m0 - off : m1 - off].T

    list(pool.map(block, range(0, K, Bin)))
    return out


# [EOF]
