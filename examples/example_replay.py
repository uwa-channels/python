import numpy as np
import scipy.signal as sg
import h5py
import matplotlib.pyplot as plt
from uwa_replay import replay, generate_noise


if __name__ == "__main__":

    channel = h5py.File("blue_1.mat", "r")
    noise = h5py.File("blue_1_noise.mat", "r")

    ## Parameters
    fs = 48e3
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
    # output = replay(input, fs, array_index, channel, start=1000)

    ## Add the noise
    output += 0.05 * generate_noise(output.shape, fs)
    # output += 0.05 * generate_noise(output.shape, fs, array_index, noise)

    ## Downconvert
    v = output * np.exp(-2j * np.pi * fc * np.arange(output.shape[0])[:, None] / fs)

    ## Plot the time domain signal
    plt.figure()
    plt.plot(np.arange(output.shape[0]) / fs, output)
    plt.xlabel("Time [s]")
    plt.ylabel("Received signal")

    ## Plot the correlation
    plt.figure()
    plt.plot(np.abs(np.correlate(v[:, 0], sg.resample_poly(data_symbols[:128], fs / R, 1), "full")))
    plt.xlabel("Samples")
    plt.ylabel("Xcorr")

    ## Plot the Welch spectrum
    plt.figure()
    plt.psd(output[:, 0], NFFT=8192, Fs=fs)
    plt.xlim(np.array([fc - R, fc + R]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/frequency (dB/Hz)")
    plt.grid()
    plt.title("Welch Power Spectral Density Estimate")

    plt.show()

# [EOF]
