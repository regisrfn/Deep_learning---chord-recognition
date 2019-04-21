import pandas as pd
import dataframe_chord

df = dataframe_chord.create_dataframe(test=True)
df.to_csv('chords_test.csv',index=False)

df = dataframe_chord.create_dataframe(test=False)
df.to_csv('chords.csv',index=False)

