[![CI](https://github.com/uwa-channels/python/actions/workflows/ci.yaml/badge.svg)](https://github.com/uwa-channels/python/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/uwa-channels/python/graph/badge.svg?token=0VK4040WNU)](https://codecov.io/gh/uwa-channels/python)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21287414-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.21287414)
[![PyPI Version](https://img.shields.io/pypi/v/uwa_channels?labelColor=green&color=blue)](https://pypi.org/project/uwa-channels/)


# Underwater Acoustic Channel Toolbox — Python

[![Generic badge](https://img.shields.io/badge/Python-3.10-BLUE.svg)](https://shields.io/)

Python toolbox for replaying signals through measured underwater acoustic channels, generating realistic ocean noise, and unpacking stored impulse responses.  To learn more about the channels, check out the [documentation](https://uwa-channels.github.io/).

Please report bugs and suggest enhancements by [creating a new issue](https://github.com/uwa-channels/python/issues).  We welcome your feedback.  See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Installation

```bash
pip install uwa-channels
```

## Functions

| Function | Description |
|----------|-------------|
| `replay` | Pass a passband signal through a measured underwater acoustic channel. |
| `noisegen` | Generate realistic ocean noise: pink Gaussian (17 dB/decade), spatially-correlated Gaussian, or impulsive (symmetric α-stable). |
| `unpack` | Reconstruct the full time-varying impulse response from the compressed representation. |
| `load_channel`, `load_noise` | Open a channel or noise MAT-file saved either with or without `-v7.3`. |

## Quick start

Download the channel MAT-files from [Zenodo](https://doi.org/10.5281/zenodo.21287414) and place them in your working directory.

### Replay and noise generation

```python
from uwa_channels import replay, noisegen, load_channel, load_noise

channel = load_channel("blue_1.mat")
noise = load_noise("blue_noise.mat")

array_index = [0, 1, 2]
y = replay(input, fs, array_index, channel)
w = noisegen(y.shape, fs, array_index, noise)
r = y + 0.05 * w
```

See `examples/example_replay.py` for a complete example that generates a BPSK signal, replays it through the `blue_1` channel, adds noise, and plots the received signal, cross-correlation, and spectrum.

> **Use `load_channel` / `load_noise` rather than `h5py.File` directly.**
> The file-format specification asks for `-v7.3` (HDF5), and `h5py` reads only
> that; MATLAB's `save` writes MAT v5 unless told otherwise. The loaders accept
> either flavour and return something `replay`, `noisegen` and `unpack` take
> unchanged — for a `-v7.3` file they return the same open `h5py.File` as
> before.

### Unpack

```python
from uwa_channels import unpack, load_channel

channel = load_channel("blue_1.mat")
unpacked = unpack(fs_time, array_index, channel)
```

See `examples/example_unpack.py` for details.

## Channel format

Each channel MAT-file contains:

| Variable | Description |
|----------|-------------|
| `h_hat` | Estimated impulse response, shape (K, M, T) |
| `theta_hat` or `phi_hat` | Phase or delay-phase trajectory, shape (M, N) |
| `params` | Group with `fs_delay`, `fs_time`, `fc` |
| `meta` | Estimation metadata (see [estimate](https://github.com/uwa-channels/estimate) repo) |
| `version` | File format version |

Each noise MAT-file contains:

| Field | Description |
|-------|-------------|
| `Fs` | Sampling rate at which noise statistics were measured [Hz] |
| `R` | Signal bandwidth [Hz] |
| `alpha` | Stability index (2 = Gaussian, < 2 = impulsive) |
| `beta` | Mixing coefficients, shape (M, M, K) |
| `fc` | Center frequency [Hz] |
| `rms_power` | Per-channel RMS power scaling, shape (M, 1) |
| `version` | Noise struct version |

## Tests

This repository includes automated testing via [GitHub Actions](https://github.com/uwa-channels/python/actions).  The [tests](/tests) folder contains five test suites:

| Test | What it verifies |
|------|-----------------|
| `test_replay` | Generates random mobile channels ({static, mobile} × {theta\_hat, phi\_hat}), transmits a signal, and checks that cross-correlation peaks match the true multipath structure; also checks that `array_index` isolates individual hydrophones and that the output power scaling is O(1). |
| `test_noisegen` | Verifies output size, spectral shape (17 dB/decade), spatial correlation (theoretical vs. sample), bandpass filtering, rms\_power scaling, Gaussianity (α = 2), and heavy-tail behavior (α < 2). |
| `test_unpack` | Tests all tracking modes (none, theta\_hat, phi\_hat, f\_resamp, and combinations) for correct impulse response reconstruction. |
| `test_io` | Writes the same channel and noise as both `-v7.3` and MAT v5, and checks that `load_channel` and `load_noise` hand `replay` and `noisegen` identical data either way. |
| `test_zenodo` | Downloads a channel and its noise file from [Zenodo](https://doi.org/10.5281/zenodo.21287414) and replays a probe through them, checking the arrival against `h_hat` and `phi_hat` and the generated noise against `beta`.  See below. |

Tests run automatically on every push, ensuring continued correctness of the core functions.

### Testing against the released files

The first four suites build their channels in memory, so they cannot catch a disagreement between the package and the files you actually download.  `test_zenodo` starts from the files: it fetches `black.mat` and `black_noise.mat` (162 MB, checked against the MD5 digests Zenodo publishes), replays a BPSK probe through the channel, and checks that

* the arrival lands within two samples of `fs_delay` of where `h_hat` and the delay drift `phi_hat / (2π f_c)` say it should,
* the measured power-delay profile matches the one stored in `h_hat`,
* the generated noise has the spatial covariance `beta` prescribes, and
* both `load_channel` and `load_noise` read the released data identically from a `-v7.3` and a MAT v5 copy.

Because of the download it is opt-in:

```bash
PYTHONPATH=src pytest tests/test_zenodo.py --zenodo
```

Files land in `~/.cache/uwa-channels`, or in `$UWA_CHANNELS_CACHE` if you set it; point that at a directory where you already keep the library and nothing is downloaded.  Once the files are there the suite runs as part of a plain `pytest`.  `UWA_CHANNELS_ZENODO=1` is equivalent to `--zenodo`.

## Related repositories

| Repository | Description |
|------------|-------------|
| [uwa-channels/matlab](https://github.com/uwa-channels/matlab) | MATLAB/Octave implementation of the replay toolbox. |
| [uwa-channels/estimate](https://github.com/uwa-channels/estimate) | Channel estimation from single-carrier signals, with visualization. |

## License

The license is available in the [LICENSE](LICENSE) file within this repository.

© 2025–2026, Underwater Acoustic Channels Group.
