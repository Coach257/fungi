import pandas as pd

s3 = pd.read_csv('S3.csv')
s4 = pd.read_csv('S4.csv')
s = pd.merge(s3,s4)
s.to_csv('S_merge3_4.csv',index = False)