import matplotlib.pyplot as plt
import numpy as np
import chord.fft_audio as fft_audio
from chord.muscial_notes import MuscialNotes
from scipy.signal import find_peaks

(freqs, Y, t, y) = fft_audio.fft("./Dataset/C-MAJOR/c-major-1.wav")

Y = Y/Y.max()
index_peaks, _ = find_peaks(Y, distance=10, height=0.1)

freqs = freqs[index_peaks]
Y = Y[index_peaks]
notes = MuscialNotes()
freqs = notes.get_muscial_notes(freqs)

# [0, 1, .. 10,11]
acorde = np.zeros(12)
for note in range(0,12):
    max_peak_index = np.where(freqs == note)[0] 
    if len(max_peak_index):
        acorde[note] = np.max(Y[max_peak_index])

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].stem(freqs, Y)  # plotting the spectrum
ax[1].set_xlabel('Freq (Musical Notes)')
ax[1].set_ylabel('|Y(freq)|')
ax[1].set_xlim(-1, 12)
plt.show()
