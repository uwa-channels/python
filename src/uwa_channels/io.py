"""Loaders that open a library ``.mat`` file whatever it was saved with.

The file-format specification asks for ``-v7.3`` (HDF5), which is what ``h5py``
reads and what the rest of the package is written against.  MAT v5 files turn
up anyway -- MATLAB's ``save`` writes v5 unless told otherwise -- and ``h5py``
rejects them with a bare ``OSError: file signature not found``.

:func:`load_noise` and :func:`load_channel` accept either flavour and return
something the rest of the package consumes unchanged.  The awkward part of the
v5 path is that HDF5 stores MATLAB arrays with their axes reversed and splits
complex values into a compound ``('real', 'imag')`` dtype, while
:func:`scipy.io.loadmat` does neither, so the fallback has to reproduce both
conventions.

Author: Zhengnan Li
Email : uwa-channels@ofdm.link

License: MIT

Revision history:
  - Sep. 15, 2026: Initial release.
"""

import numpy as np
import h5py

__all__ = ["load_noise", "load_channel", "is_hdf5"]


def is_hdf5(path):
    """True if ``path`` is an HDF5 (``-v7.3``) MAT-file.

    A MATLAB ``-v7.3`` file is HDF5 behind a 512-byte userblock holding the
    ``MATLAB 7.3 MAT-file ...`` banner, so the HDF5 signature is not at offset
    zero and a naive magic-number check on the first eight bytes says "no" for
    every file in the library.  :func:`h5py.is_hdf5` knows about the userblock.
    """
    try:
        return bool(h5py.is_hdf5(str(path)))
    except (OSError, ValueError):
        return False


class _ComplexCompound:
    """Present a native complex array the way ``h5py`` presents a MATLAB one.

    ``h5py`` returns complex MATLAB data as a compound dataset, so the package
    reads ``dataset["real"]`` and ``dataset["imag"]``.  ``scipy.io.loadmat``
    returns a plain complex array instead; this wrapper gives it the same two
    keys so callers need no special case.
    """

    __slots__ = ("_z",)

    def __init__(self, z):
        self._z = np.asarray(z)

    def __getitem__(self, key):
        if key == "real":
            return self._z.real
        if key == "imag":
            return self._z.imag
        raise KeyError(
            f"{key!r}; a complex MAT variable exposes only 'real' and 'imag'."
        )

    @property
    def shape(self):
        return self._z.shape

    @property
    def ndim(self):
        return self._z.ndim

    @property
    def dtype(self):
        return self._z.dtype

    def __array__(self, dtype=None):
        return np.asarray(self._z, dtype=dtype)


def _struct_to_dict(value):
    """Turn a ``scipy.io.loadmat`` struct into a dict of 2-D arrays.

    ``loadmat`` returns a MATLAB struct as a 1x1 structured array whose fields
    are themselves arrays.  The package reads scalars as ``group[name][0, 0]``,
    matching how ``h5py`` presents them, so every field is returned at least
    2-D.
    """
    rec = np.asarray(value)
    while rec.dtype == object and rec.size == 1:
        rec = rec.reshape(-1)[0]
    if rec.dtype.names is None:
        return None
    rec = rec.reshape(-1)[0] if rec.size == 1 else rec
    out = {}
    for name in rec.dtype.names:
        field = rec[name]
        field = np.asarray(field)
        while field.dtype == object and field.size == 1:
            field = field.reshape(-1)[0]
            field = np.asarray(field)
        out[name] = np.atleast_2d(field)
    return out


def _loadmat_v5(path):
    """Read a MAT v5 file, dropping the ``__header__``-style entries."""
    from scipy.io import loadmat

    raw = loadmat(path, squeeze_me=False, struct_as_record=True)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def load_noise(path):
    """Open a noise ``.mat`` file saved either with or without ``-v7.3``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a noise MAT-file, for example ``blue_noise.mat`` or
        ``red_noise.mat``.

    Returns
    -------
    mapping
        For a ``-v7.3`` file, the open :class:`h5py.File` -- identical to what
        ``h5py.File(path, "r")`` would have returned, so nothing about existing
        code changes.  For a MAT v5 file, a plain dict of arrays.  Either can be
        passed straight to :func:`~uwa_channels.noisegen`.

    Notes
    -----
    ``noisegen`` accepts both layouts of ``beta``: ``h5py`` reports it as
    ``(L+1, M, M)`` because HDF5 reverses the MATLAB axes, ``loadmat`` reports
    the stored ``(M, M, L+1)``, and the mixing code normalizes whichever it is
    given.  No transposition is applied here.

    Examples
    --------
    >>> from uwa_channels import load_noise, noisegen      # doctest: +SKIP
    >>> noise = load_noise("blue_noise.mat")               # doctest: +SKIP
    >>> w = noisegen((4096, 3), 48e3, [0, 1, 2], noise)    # doctest: +SKIP
    """
    if is_hdf5(path):
        return h5py.File(path, "r")
    return _loadmat_v5(path)


def load_channel(path):
    """Open a channel ``.mat`` file saved either with or without ``-v7.3``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a channel MAT-file, for example ``blue_1.mat``.

    Returns
    -------
    mapping
        For a ``-v7.3`` file, the open :class:`h5py.File`.  For a MAT v5 file, a
        dict shaped to match what ``h5py`` would have returned: ``h_hat`` is
        exposed with ``"real"`` and ``"imag"`` keys, and the axes of ``h_hat``,
        ``phi_hat`` and ``theta_hat`` are reversed.  Either can be passed
        straight to :func:`~uwa_channels.replay` or
        :func:`~uwa_channels.unpack`.

    Notes
    -----
    The fallback matters most for channels you estimate yourself: MATLAB's
    ``save`` writes v5 unless told otherwise, and the resulting file is
    perfectly good data that ``h5py`` simply cannot read.  Save with
    ``-v7.3 -nocompression`` when you can, as the file-format specification
    asks.

    Examples
    --------
    >>> from uwa_channels import load_channel, replay      # doctest: +SKIP
    >>> channel = load_channel("blue_1.mat")               # doctest: +SKIP
    >>> y = replay(x, 48e3, [0, 1, 2], channel)            # doctest: +SKIP
    """
    if is_hdf5(path):
        return h5py.File(path, "r")

    raw = _loadmat_v5(path)
    out = {}
    for key, value in raw.items():
        if key == "h_hat":
            z = np.asarray(value)
            # MATLAB stores [delay, receiver, time]; h5py reports the reverse.
            out[key] = _ComplexCompound(np.transpose(z, tuple(reversed(range(z.ndim)))))
        elif key in ("phi_hat", "theta_hat"):
            out[key] = np.asarray(value).T
        elif key in ("params", "meta"):
            fields = _struct_to_dict(value)
            out[key] = fields if fields is not None else value
        else:
            out[key] = np.atleast_2d(np.asarray(value))
    return out


# [EOF]
