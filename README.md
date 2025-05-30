[![CI](https://github.com/uwa-channels/python/actions/workflows/ci.yaml/badge.svg)](https://github.com/uwa-channels/python/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/uwa-channels/python/graph/badge.svg?token=0VK4040WNU)](https://codecov.io/gh/uwa-channels/python)

# Underwater Acoustic Channel Toolbox - Python

[![Generic badge](https://img.shields.io/badge/Python-3.10-BLUE.svg)](https://shields.io/)

Python toolbox to apply underwater acoustic channels to a signal of your choice, or to unpack an underwater acoustic channel. To learn more about the channels, check out the [documentation](https://uwa-channels.github.io/).

Please report bugs and suggest enhancements by [creating a new issue](https://github.com/uwa-channels/python/issues). We welcome your comments. See [CONTRIBUTING.MD](CONTRIBUTING.md) for more information.

To install the toolbox,

```bash
pip install -i https://test.pypi.org/simple/ uwa-replay
```

## Using the replay and noise generation functions

This code repository contains the Python function `replay` and noise generation function `noisegen`. To replay your desired signal, download MAT-files from [here](https://www.dropbox.com/scl/fo/3gyt4cgw47jfx716v0epd/AIqYaL5S2RxGylREu3sn-vY?rlkey=w2mvoklkm42zrrf6k6lwlzcxu&st=u3u6b5r9&dl=0), and store them in a folder where Python can find them.

To load the channel and noise MAT-files, and replay a signal of your choice through an underwater acoustic channel,
```python
from uwa_replay import replay, noisegen
channel = h5py.File("blue_1.mat", "r")
noise = h5py.File("blue_1_noise.mat", "r")
y = replay(input, fs, array_index, channel)
w = noisegen(y.shape, fs, array_index, noise)
r = y + 0.05 * w
```

In `examples/example_replay.py`, the `blue_1` channel is used. The `blue_1.mat` contains the channel impulse responses, while the `blue_1_noise.mat` contains the noise statistics extracted from the same recording. The script generates a single-carrier modulated BPSK signal consisting of `n_repeat` repetitions of a pseudo-random sequence, passes the signal through the `blue_1` channel, and adds `blue_1_noise` noise. Three plots are displayed: the received signal amplitude in time, the cross-correlation between the received signal and the transmitted signal, where `n_repeat` peaks are visible, and the spectrum of the received signal. Multiple curves on each plot correspond to multiple receiving elements.

## Using the unpack function

To unpack an underwater acoustic channel,
```python
from uwa_replay import unpack
channel = h5py.load('blue_1.mat');
unpacked = unpack(fs_time, array_index, channel);
```

See `examples/example_unpack.py` for details.

## Tests
This repository includes automated testing and deployment powered by [GitHub Actions](https://github.com/uwa-channels/python/actions). In the [tests](/tests) folder, you will find three test suites covering the core functionalities: replay, noise generation, and unpacking.

In particular, the replay test suite generates a random mobile channel and transmits a signal through it. A simple matched filter is then applied to verify whether the correlation peaks correspond to the actual channel multipath structure. If specific criteria are met, the test passes.

These tests are executed automatically whenever changes are made to the source code, ensuring the continued correctness of the core functionalities.

# License
The license is available in the [LICENSE](LICENSE) file within this repository.

© 2025, Underwater Acoustic Channels Group.
