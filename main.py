import pandas as pd
from dataframe_chord import DataFrame

df = DataFrame(300,folder='./Dataset/Test/')
test_df = df.create_dataframe()
test_df.to_csv('chords_test.csv',index=False)

# df = DataFrame(200,folder='./Dataset/Main/')
# test_df = df.create_dataframe()
# test_df.to_csv('chords.csv',index=False)

