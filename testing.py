import numpy as np
import pandas as pd #en las consignas dice solo numpy, pero me imagino q para cargar el df se puede usar pandas

#https://pandas.pydata.org/docs/reference/frame.html
#https://numpy.org/doc/stable/user/index.html

df = pd.read_csv('user_behavior_dataset.csv', sep=',')

#df basics
print('Basics')
print(df.index)
print('\n')
#deleting non usable columns
print('Columns')
print(df.columns)
print('\n')
print('Erasing Columns')
del df['Data Usage (MB/day)'], df['Battery Drain (mAh/day)'], df['User ID']
print(df)
print('\n')

#matrix of values (usable for numpy)
print('Matrix of values')
print(df.values)
print('\n')

#Description of the Data
print('Describe')
print(df.describe())
print('\n')

#DataFrame to Numpy Array
print('Df To numpy')
np_array = df.to_numpy()
print(np_array)
print('\n')

#Accessing a singular position .at[row, column_name]
print('.at[row, column_name]')
print(df.at[0,'App Usage Time (min/day)'])
print('\n')
#Accessing a singular position .iat[row, column_index]
print('.iat[row, column_index]')
print(df.iat[0,2])
print('\n')
#Accesing whole row
print('whole row')
print(df.loc[0])
print('\n')
#Accessing by purely integer position
#This is a DataFrame
print('iloc dataframe')
print(df.iloc[[0]])
print('\n')
#This is a Series
print('iloc series')
print(df.iloc[0])
print('\n')