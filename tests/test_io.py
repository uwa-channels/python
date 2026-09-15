"""
Unit tests for io.py

The package reads MAT-files with h5py, which only understands the ``-v7.3``
(HDF5) flavour that the file-format specification asks for.  Files saved
without that flag are MAT v5, and h5py rejects them with a bare
``OSError: file signature not found``.

``load_noise`` and ``load_channel`` open either flavour and hand back something
``noisegen``, ``replay`` and ``unpack`` consume unchanged.  These tests write
both flavours of the same data and check that the loaders agree.

Authors: Zhengnan Li
Email  : uwa-channels@ofdm.link
License: MIT

Revision history:
  - Sep. 15, 2026: Initial release.
"""

import numpy as np
import pytest
import scipy.io as sio
import h5py

from uwa_channels import load_channel, load_noise, is_hdf5, noisegen, replay


# ---------------------------------------------------------------------------
# Fixtures: the same small noise and channel, written both ways
# ---------------------------------------------------------------------------

M, L1, K, T = 3, 9, 16, 40
FS_NOISE, FC, R = 8000.0, 2000.0, 1000.0

FS_DELAY, FS_TIME = 2 * R, R / 100          # 2000 Hz and 10 Hz
N_PHI = int(T / FS_TIME * FS_DELAY)         # same duration as h_hat: 8000


def _noise_arrays(seed=0):
    rng = np.random.default_rng(seed)
    beta = np.zeros((M, M, L1))                 # MATLAB layout [M, M, L+1]
    beta[:, :, 0] = np.eye(M) + np.triu(0.4 * np.ones((M, M)), 1)
    beta[:, :, 1:] = 0.05 * rng.standard_normal((M, M, L1 - 1))
    # Deliberately asymmetric in (i, j): a symmetric beta cannot tell a
    # correct layout from a transposed one, which is the bug this guards.
    assert not np.allclose(beta[:, :, 0], beta[:, :, 0].T)
    return beta


def _channel_arrays(seed=1):
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((K, M, T))         # MATLAB layout [delay, rx, time]
         + 1j * rng.standard_normal((K, M, T))).astype(np.complex128)
    phi = rng.standard_normal((M, N_PHI))       # MATLAB layout [rx, time]
    return h, phi


@pytest.fixture
def noise_pair(tmp_path):
    beta = _noise_arrays()
    v5 = tmp_path / "noise_v5.mat"
    v73 = tmp_path / "noise_v73.mat"

    sio.savemat(v5, {"Fs": FS_NOISE, "R": R, "alpha": 2.0, "fc": FC,
                     "beta": beta, "version": 1.0})

    # h5py writes C-order, which is how MATLAB's -v7.3 ends up looking to a
    # reader: the axes come back reversed relative to the stored MATLAB array.
    with h5py.File(v73, "w") as f:
        for name, val in (("Fs", FS_NOISE), ("R", R), ("alpha", 2.0),
                          ("fc", FC), ("version", 1.0)):
            f.create_dataset(name, data=np.array([[val]]))
        f.create_dataset("beta", data=np.transpose(beta, (2, 1, 0)))

    return v5, v73, beta


@pytest.fixture
def channel_pair(tmp_path):
    h, phi = _channel_arrays()
    v5 = tmp_path / "chan_v5.mat"
    v73 = tmp_path / "chan_v73.mat"

    params = np.zeros((1, 1), dtype=[("fs_delay", "O"), ("fs_time", "O"),
                                     ("fc", "O")])
    params["fs_delay"][0, 0] = np.array([[FS_DELAY]])
    params["fs_time"][0, 0] = np.array([[FS_TIME]])
    params["fc"][0, 0] = np.array([[FC]])
    sio.savemat(v5, {"h_hat": h, "phi_hat": phi, "params": params,
                     "version": 1.0})

    with h5py.File(v73, "w") as f:
        g = f.create_group("params")
        g.create_dataset("fs_delay", data=np.array([[FS_DELAY]]))
        g.create_dataset("fs_time", data=np.array([[FS_TIME]]))
        g.create_dataset("fc", data=np.array([[FC]]))
        f.create_dataset("version", data=np.array([[1.0]]))
        ht = np.transpose(h, (2, 1, 0))
        comp = np.zeros(ht.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
        comp["real"], comp["imag"] = ht.real, ht.imag
        f.create_dataset("h_hat", data=comp)
        f.create_dataset("phi_hat", data=phi.T)

    return v5, v73, h, phi


# ---------------------------------------------------------------------------
# is_hdf5
# ---------------------------------------------------------------------------

def test_is_hdf5_discriminates(noise_pair):
    v5, v73, _ = noise_pair
    assert is_hdf5(v73)
    assert not is_hdf5(v5)


def test_is_hdf5_missing_file(tmp_path):
    assert not is_hdf5(tmp_path / "nope.mat")


# ---------------------------------------------------------------------------
# load_noise
# ---------------------------------------------------------------------------

def test_load_noise_opens_v5(noise_pair):
    """h5py refuses a v5 file; load_noise does not."""
    v5, _, beta = noise_pair
    with pytest.raises(OSError):
        h5py.File(v5, "r")

    noise = load_noise(v5)
    assert float(np.asarray(noise["alpha"]).ravel()[0]) == 2.0
    assert np.allclose(np.asarray(noise["beta"]), beta)


def test_load_noise_v73_is_unchanged(noise_pair):
    _, v73, _ = noise_pair
    noise = load_noise(v73)
    assert isinstance(noise, h5py.File)
    noise.close()


@pytest.mark.parametrize("which", [0, 1])
def test_noisegen_accepts_either_flavour(noise_pair, which):
    """Both loaders drive noisegen, and the two agree statistically."""
    paths = noise_pair[:2]
    np.random.seed(7)
    w = noisegen((4096, M), FS_NOISE, list(range(M)), load_noise(paths[which]))
    assert w.shape == (4096, M)
    assert np.all(np.isfinite(w))
    assert np.std(w) > 0


def test_noisegen_same_stream_from_both_flavours(noise_pair):
    v5, v73, _ = noise_pair
    np.random.seed(11)
    a = noisegen((2048, M), FS_NOISE, list(range(M)), load_noise(v5))
    np.random.seed(11)
    b = noisegen((2048, M), FS_NOISE, list(range(M)), load_noise(v73))
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# load_channel
# ---------------------------------------------------------------------------

def test_load_channel_v5_matches_h5py_layout(channel_pair):
    """The v5 path must reproduce h5py's reversed axes and compound complex."""
    v5, v73, h, phi = channel_pair
    a, b = load_channel(v5), load_channel(v73)

    assert a["h_hat"].shape == b["h_hat"].shape == (T, M, K)
    assert np.allclose(np.asarray(a["h_hat"]["real"]), np.asarray(b["h_hat"]["real"]))
    assert np.allclose(np.asarray(a["h_hat"]["imag"]), np.asarray(b["h_hat"]["imag"]))
    assert np.allclose(np.asarray(a["h_hat"]["real"]), np.transpose(h, (2, 1, 0)).real)

    assert np.asarray(a["phi_hat"]).shape == np.asarray(b["phi_hat"]).shape
    assert np.allclose(np.asarray(a["phi_hat"]), phi.T)

    for key in ("fs_delay", "fs_time", "fc"):
        assert a["params"][key][0, 0] == pytest.approx(b["params"][key][0, 0])

    assert float(np.asarray(a["version"]).ravel()[0]) == 1.0
    b.close()


def test_replay_accepts_a_v5_channel(channel_pair):
    """End to end: a channel saved without -v7.3 still replays."""
    v5, v73, _, _ = channel_pair
    x = np.random.default_rng(3).standard_normal(1000)   # 0.5 s of a 4 s channel

    y5 = replay(x, FS_DELAY, [0, 1], load_channel(v5), start=5)
    ch73 = load_channel(v73)
    y73 = replay(x, FS_DELAY, [0, 1], ch73, start=5)
    ch73.close()

    assert y5.shape == y73.shape
    assert np.allclose(y5, y73)


# ---------------------------------------------------------------------------
# The mixing convention
# ---------------------------------------------------------------------------

def test_beta_layout_is_not_transposed(noise_pair):
    """The generated noise must follow Eq. (32), n_i = sum_j beta_ij eta_j.

    h5py reverses every axis of a MATLAB array, so a [M, M, L+1] beta arrives
    as (L+1, M, M).  Undoing that with the wrong permutation leaves beta
    transposed in (i, j), which silently changes the spatial covariance of the
    generated noise.  beta is asymmetric here, and asymmetric in the released
    files, so the two conventions are distinguishable.
    """
    v5, v73, beta = noise_pair

    def coherence(R):
        d = np.sqrt(np.diag(R))
        return np.abs(R / np.outer(d, d))

    spec = coherence(sum(beta[:, :, k] @ beta[:, :, k].T
                         for k in range(beta.shape[2])))
    transposed = coherence(sum(beta[:, :, k].T @ beta[:, :, k]
                               for k in range(beta.shape[2])))
    assert not np.allclose(spec, transposed), "fixture cannot tell the two apart"

    np.random.seed(19)
    w = noisegen((2 ** 18, M), FS_NOISE, list(range(M)), load_noise(v73))
    emp = coherence(np.cov(w.T))

    d_spec = np.sqrt(np.mean((emp - spec) ** 2))
    d_trans = np.sqrt(np.mean((emp - transposed) ** 2))
    assert d_spec < d_trans, (
        f"generated noise matches the transposed mixing convention "
        f"(rms {d_trans:.4f}) better than Eq. (32) (rms {d_spec:.4f})"
    )


def test_both_flavours_give_the_same_covariance(noise_pair):
    """A v5 and a -v7.3 copy of one beta must generate the same noise."""
    v5, v73, _ = noise_pair
    np.random.seed(23)
    a = noisegen((2 ** 16, M), FS_NOISE, list(range(M)), load_noise(v5))
    np.random.seed(23)
    b = noisegen((2 ** 16, M), FS_NOISE, list(range(M)), load_noise(v73))
    assert np.allclose(a, b)

# [EOF]
