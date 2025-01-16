import numpy as np
import scipy.signal as sg
import h5py
import matplotlib.pyplot as plt
from replay import replay
from generate_noise import generate_noise
from generate_impulsive_noise import generate_impulsive_noise


if __name__ == "__main__":

    channel = h5py.File("blue_1.mat", "r")
    noise = h5py.File("blue_1_noise.mat", "r")

    ## Parameters
    fs = 96e3
    fc = 13e3
    R = 4e3
    n_repeat = 10
    array_index = np.array([0, 2, 4])

    ## Generate single carrier signals
    data_symbols = np.random.choice([-1.0, +1.0], size=(1023,))
    baseband = sg.resample_poly(np.tile(data_symbols, n_repeat), fs / R, 1)
    passband = np.real(baseband * np.exp(2j * np.pi * fc * np.arange(len(baseband)) / fs))
    input = np.concatenate((np.zeros((int(fs / 10),)), passband, np.zeros((int(fs / 10)))))

    ## Replay and generate noise
    output = replay(input, fs, array_index, channel)

    ## Add the noise
    output += 0.05 * generate_noise(output, fs, array_index, noise, 3)
    # output += 0.05 * generate_impulsive_noise(output, fs, array_index, noise)

    ## Downconvert
    v = output * np.exp(-2j * np.pi * fc * np.arange(output.shape[0])[:, None] / fs)

    ## Plot the correlation
    plt.figure()
    plt.plot(np.abs(np.correlate(v[:, 0], sg.resample_poly(data_symbols[:128], fs / R, 1), "full")))
    plt.xlabel("Samples")
    plt.ylabel("Xcorr")

    ## Plot the Welch spectrum
    plt.figure()
    f, pxx_den = sg.welch(
        output[:, 0], fs, detrend=False, window=sg.get_window(("kaiser", 0.5), 1024), noverlap=512, nfft=4096
    )
    plt.plot(f, 10 * np.log10(pxx_den))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power/frequency (dB/Hz)")
    plt.title("Welch Power Spectral Density Estimate")

    plt.show()

# [EOF]
