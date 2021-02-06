import pandas as pd

s = pd.read_csv('info.csv')
maxwidth = 0.0
minwidth = 5.0
for index,row in s.iterrows():
    maxwidth = max(maxwidth,row['water.niche.width'])
    minwidth = min(minwidth,row['water.niche.width'])
for index,row in s.iterrows():
    s.loc[index, 'water.niche.width'] = (row['water.niche.width'] - minwidth) / (maxwidth - minwidth)
s.to_csv('info_width_to_0-1.csv')