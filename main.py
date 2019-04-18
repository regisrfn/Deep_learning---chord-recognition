import matplotlib.pyplot as plt
import numpy as np
import chord.fft_audio as fft_audio
from chord.muscial_notes import MuscialNotes
from scipy.signal import find_peaks

(freqs, Y, t, y) = fft_audio.fft("./Dataset/C-MAJOR/c-major-1.wav")

index_peaks, _ = find_peaks(Y, distance=10, height=10)

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(freqs, Y, 'r')  # plotting the spectrum
ax[1].plot(freqs[index_peaks], Y[index_peaks], 'x')  # plotting the spectrum
ax[1].set_xlabel('Freq (Musical Notes)')
ax[1].set_ylabel('|Y(freq)|')
ax[1].set_xlim(0, 1000)
plt.show()
