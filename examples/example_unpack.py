import numpy as np
import h5py
import matplotlib.pyplot as plt
from uwa_replay import unpack


if __name__ == "__main__":
    ## Load channel impulse responses
    channel = h5py.File("blue_1.mat", "r")

    ## Parammeters
    fs_delay = channel["params"]["fs_delay"][0, 0]
    fs_time = 20
    array_index = [0, 2, 4]

    ## Unpack the channel
    unpacked = unpack(fs_time, array_index, channel)

    ## Visualize
    delay_axis = np.arange(unpacked.shape[0]) / fs_delay
    time_axis = np.arange(unpacked.shape[2]) / fs_time
    plt.pcolor(
        delay_axis * 1e3,
        time_axis,
        20 * np.log10(np.abs(np.squeeze(unpacked[:, 0, :]))).T,
        vmin=-30,
        vmax=0,
    )
    plt.gca().invert_yaxis()  # Align with MATLAB
    plt.xlabel("Delay [ms]")
    plt.ylabel("Time [s]")

    plt.show()

# [EOF]
