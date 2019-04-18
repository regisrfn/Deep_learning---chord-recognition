import numpy as np
from scipy.io import wavfile

def fft(filename):

    samplerate, data = wavfile.read(filename)

    Fs = samplerate  # sampling rate
    Ts = 1.0/Fs  # sampling interval
    t = np.arange(len(data)) * Ts  # time vector

    channels = len(data.shape)
    # convert to mono by averaging the left and right channels.
    if(channels > 1):
        y = np.mean(data, axis=1)
    else:
        y = data

    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n/Fs
    freqs = k/T  # two sides frequency range
    freqs = freqs[:int(n/2)]  # one side frequency range

    Y = np.fft.fft(y)/n  # fft computing and normalization
    Y = abs(Y[:int(n/2)])

    return (freqs, Y , t, y)
