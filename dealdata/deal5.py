import pandas as pd

df = pd.read_csv('new_info_clean.csv')

for index,row in df.iterrows():
    str1 = row['Extension rate 16 ']
    str2 = row['Decomposition rate 16 ']
    str1 = str1[0:str1.index("±")]
    str2 = str2[0:str2.index("±")]
    df.loc[index,'Extension rate 16 '] = float(str1)
    df.loc[index,'Decomposition rate 16 '] = float(str2)

df.to_csv('New_info_clean.csv',index=False)