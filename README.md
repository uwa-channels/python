[![CI](https://github.com/uwa-channels/replay_python/actions/workflows/ci.yaml/badge.svg)](https://github.com/uwa-channels/replay_python/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/uwa-channels/replay_python/graph/badge.svg?token=0VK4040WNU)](https://codecov.io/gh/uwa-channels/replay_python)

# Underwater Acoustic Channel Toolbox - Python

[![Generic badge](https://img.shields.io/badge/Python-3.10-BLUE.svg)](https://shields.io/)

Python toolbox to apply underwater acoustic channels to a signal of your choice, or to unpack an underwater acoustic channel. To learn more about the channels, check out the [documentation](https://uwa-channels.github.io/). 

Please report bugs and suggest enhancements by [creating a new issue](https://github.com/uwa-channels/replay_python/issues). We welcome your comments. See [CONTRIBUTING.MD](CONTRIBUTING.md) for more information.

## Using the replay toolbox

This code repository contains the Python function `replay` and noise generation functions `generate_noise` and `generate_impulsive_noise`. To replay your desired signal, download MAT-files from [here](https://www.dropbox.com/scl/fo/3gyt4cgw47jfx716v0epd/AIqYaL5S2RxGylREu3sn-vY?rlkey=w2mvoklkm42zrrf6k6lwlzcxu&st=u3u6b5r9&dl=0), and store them in a folder where Python can find them.

To install the requirements,

```bash
pip install replay_python
```

To load the channel and noise MAT-files, and replay a signal of your choice through an underwater acoustic channel,
```python
channel = h5py.load("blue_1.mat")
noise = h5py.load("blue_1_noise.mat")
y = replay(input, fs, array_index, channel)
w = generate_noise(y.shape, fs)
r = y + 0.05 * w
```

In `examples/example_replay.py`, the `blue_1` channel is used. The `blue_1.mat` contains the channel impulse responses, while the `blue_1_noise.mat` contains the noise statistics extracted from the same recording. The script generates a single-carrier modulated BPSK signal consisting of `n_repeat` repetitions of a pseudo-random sequence, passes the signal through the `blue_1` channel, and adds `blue_1` noise. Three plots are displayed: the received signal amplitude in time, the cross-correlation between the received signal and the transmitted signal, where `n_repeat` peaks are visible, and the spectrum of the received signal. Multiple curves on each plot correspond to multiple receiving elements.

Note that there are two noise generator functions: `generate_noise` and `generate_impulsive_noise`. The `generate_noise` should be used when working with either `blue`, `green`, `purple`, or `yellow` channel, while `generate_impulsive_noise` requires the `red_noise.mat` to be loaded.

## Using the unpack function

To unpack an underwater acoustic channel,
```python
channel = h5py.load('blue_1.mat');
unpacked = unpack(fs_time, array_index, channel);
```

See `examples/example_unpack.py` for details.

# License
The license is available in the [LICENSE](LICENSE) file within this repository.

© 2025, Underwater Acoustic Channels Group.
