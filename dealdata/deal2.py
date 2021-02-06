import pandas as pd

s1 = pd.read_csv('S_merge3_4.csv')
s2 = pd.read_csv('Fungal_trait_data.csv')
s2 = s2.drop(s2.columns[0],axis = 1)
for index,row in s1.iterrows():
    str = row['gen.name2']
    if (str == "Phellinus robiniae AZ15 A10H Banik/Mark") :
        str = "Phellinus_robiniae_AZ15_A10H Banik/Mark"
    else :
        str = str.replace(" ","_")
    s1.loc[index,'gen.name2'] = str
s = pd.merge(s1,s2)
s.to_csv('info.csv',index = False)

