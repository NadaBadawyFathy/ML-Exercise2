import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv('hiring.csv')
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(math.floor(df['test_score(out of 10)'].mean()))

df.experience = df.experience.fillna('zero')

df.experience = df.experience.apply(w2n.word_to_num)

model = linear_model.LinearRegression()

model.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

print(df)
print(model.predict([[2,9,6]]))
print(model.predict([[12,10,10]]))