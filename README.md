# Replay - Python

Python toolbox to apply underwater acoustic channels to a signal of your choice. To learn more about the channels, check out the [documentation](https://uwa-channels.github.io/). 

Please report bugs and suggest enhancements by [creating a new issue](https://github.com/uwa-channels/replay_python/issues). We welcome your comments. See [CONTRIBUTING.MD](CONTRIBUTING.md) for more information.

## Using the replay toolbox

This code repository contains the Python function `replay` and noise generation functions `generate_noise` and `generate_impulsive_noise`. To replay your desired signal, download MAT-files from [here](https://www.dropbox.com/scl/fo/3gyt4cgw47jfx716v0epd/AIqYaL5S2RxGylREu3sn-vY?rlkey=w2mvoklkm42zrrf6k6lwlzcxu&st=u3u6b5r9&dl=0), and store them in a folder where Python can find them.

To install the requirements,

```bash
pip install -r requirements.txt
```

In `example.py`, the `blue_1` channel is used. The `blue_1.mat` contains the channel impulse responses, and the `blue_1_noise.mat` contains the noise statistics extracted from the same recording. The script generates a single-carrier modulated BPSK signal consisting of `n_repeat` repetitions of a pseudo-random sequence, passes the signal through the `blue_1` channel, and adds noise. Two plots are displayed, one showing the cross-correlation between the received signal and the transmitted signal, where `n_repeat` peaks are visible, and another showing the spectrum of the received signal.

Note that the `generate_impulsive_noise` function requires the `red_noise.mat` to be loaded.

# License
The license is available in the [LICENSE](LICENSE) file within this repository.

© 2025, Underwater Acoustic Channels Group.
