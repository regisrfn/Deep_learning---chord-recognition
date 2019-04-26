import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chord.fft_audio as fft_audio
from chord.muscial_notes import MuscialNotes
from scipy.signal import find_peaks


class DataFrame:
    def __init__(self,size,folder):
        self.size =  size
        self.folder = folder
        self.chords = ['c/c', 'd/d', 'dm/dm', 'e/e', 'g/g']


    def create_dataframe(self):
        table = []

        size = self.size
        folder = self.folder
        names_chords = self.chords

        for (index, chord_name) in enumerate(names_chords):
            for i in range(size):
                audiofile = f'{folder}{chord_name}{i+1}.wav'
                (freqs, Y, _, _) = fft_audio.fft(audiofile)
                Y = Y/Y.max()
                Y = 20 * np.log10(Y)
                index_peaks, _ = find_peaks(Y, distance=1)

                freqs = freqs[index_peaks]
                Y = Y[index_peaks]
                notes = MuscialNotes()
                freqs = notes.get_muscial_notes(freqs)

                # [0, 1, .. 10,11]
                acorde = np.zeros(13)
                for note in range(0, 12):
                    max_peak_index = np.where(freqs == note)[0]
                    if len(max_peak_index):
                        acorde[note] = np.max(Y[max_peak_index])

                acorde[12] = index
                table.append(acorde)

        # Create the pandas DataFrame
        df = pd.DataFrame(table, columns=[
                          'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'Acorde'])

        return df
        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(t, y)
        # ax[0].set_xlabel('Time')
        # ax[0].set_ylabel('Amplitude')
        # ax[1].stem(freqs, Y)  # plotting the spectrum
        # ax[1].set_xlabel('Freq (Musical Notes)')
        # ax[1].set_ylabel('|Y(freq)|')
        # ax[1].set_xlim(-1, 12)
        # plt.show()
