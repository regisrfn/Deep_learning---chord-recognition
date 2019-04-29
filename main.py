import pandas as pd
from dataframe_chord import DataFrame

df = DataFrame(10,folder='./Dataset/Test/Accordion/')
test_df = df.create_dataframe()
test_df.to_csv('./dataset_csv/chords_test_accordion.csv',index=False)

# df = DataFrame(200,folder='./Dataset/Main/')
# test_df = df.create_dataframe()
# test_df.to_csv('./dataset_csv/chords.csv',index=False)


