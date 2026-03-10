[![CI](https://github.com/uwa-channels/python/actions/workflows/ci.yaml/badge.svg)](https://github.com/uwa-channels/python/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/uwa-channels/python/graph/badge.svg?token=0VK4040WNU)](https://codecov.io/gh/uwa-channels/python)

# Underwater Acoustic Channel Toolbox — Python

[![Generic badge](https://img.shields.io/badge/Python-3.10-BLUE.svg)](https://shields.io/)

Python toolbox for replaying signals through measured underwater acoustic channels, generating realistic ocean noise, and unpacking stored impulse responses.  To learn more about the channels, check out the [documentation](https://uwa-channels.github.io/).

Please report bugs and suggest enhancements by [creating a new issue](https://github.com/uwa-channels/python/issues).  We welcome your feedback.  See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Installation

```bash
pip install -i https://test.pypi.org/simple/ uwa-channels
```

## Functions

| Function | Description |
|----------|-------------|
| `replay` | Pass a passband signal through a measured underwater acoustic channel. |
| `noisegen` | Generate realistic ocean noise: pink Gaussian (17 dB/decade), spatially-correlated Gaussian, or impulsive (symmetric α-stable). |
| `unpack` | Reconstruct the full time-varying impulse response from the compressed representation. |

## Quick start

Download the channel MAT-files from [here](https://www.dropbox.com/scl/fo/3gyt4cgw47jfx716v0epd/AIqYaL5S2RxGylREu3sn-vY?rlkey=w2mvoklkm42zrrf6k6lwlzcxu&st=u3u6b5r9&dl=0) and place them in your working directory.

### Replay and noise generation

```python
import h5py
from uwa_channels import replay, noisegen

channel = h5py.File("blue_1.mat", "r")
noise = h5py.File("blue_1_noise.mat", "r")

y = replay(input, fs, array_index, channel)
w = noisegen(y.shape, fs, array_index, noise)
r = y + 0.05 * w
```

See `examples/example_replay.py` for a complete example that generates a BPSK signal, replays it through the `blue_1` channel, adds noise, and plots the received signal, cross-correlation, and spectrum.

### Unpack

```python
import h5py
from uwa_channels import unpack

channel = h5py.File("blue_1.mat", "r")
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

This repository includes automated testing via [GitHub Actions](https://github.com/uwa-channels/python/actions).  The [tests](/tests) folder contains three test suites:

| Test | What it verifies |
|------|-----------------|
| `test_replay` | Generates random mobile channels ({static, mobile} × {theta\_hat, phi\_hat}), transmits a signal, and checks that cross-correlation peaks match the true multipath structure. |
| `test_noisegen` | Verifies output size, spectral shape (17 dB/decade), spatial correlation (theoretical vs. sample), bandpass filtering, rms\_power scaling, Gaussianity (α = 2), and heavy-tail behavior (α < 2). |
| `test_unpack` | Tests all tracking modes (none, theta\_hat, phi\_hat, f\_resamp, and combinations) for correct impulse response reconstruction. |

Tests run automatically on every push, ensuring continued correctness of the core functions.

## Related repositories

| Repository | Description |
|------------|-------------|
| [uwa-channels/matlab](https://github.com/uwa-channels/matlab) | MATLAB/Octave implementation of the replay toolbox. |
| [uwa-channels/estimate](https://github.com/uwa-channels/estimate) | Channel estimation from single-carrier signals, with visualization. |

## License

The license is available in the [LICENSE](LICENSE) file within this repository.

© 2025–2026, Underwater Acoustic Channels Group.
