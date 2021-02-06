import pandas as pd

s = pd.read_csv('info_clean.csv')
for index,row in s.iterrows():
    s.loc[index, 'my_tolerance'] = row['ranking'] - row['water.niche.width']
s.to_csv('new_info_clean.csv',index=False)