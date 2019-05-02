import pandas as pd
from dataframe_chord import DataFrame

df = DataFrame(300,folder='./Dataset/Test/')
df.chords = df.chords[:5]
test_df = df.create_dataframe()
test_df.to_csv('./dataset_csv/chords_test_2.csv',index=False)

df = DataFrame(200,folder='./Dataset/Main/')
test_df = df.create_dataframe()
test_df.to_csv('./dataset_csv/chords_2.csv',index=False)


