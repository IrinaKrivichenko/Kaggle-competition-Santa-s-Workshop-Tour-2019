import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")
def set_days_125():
    days_125 = set()
    d_125 = 8
    for _ in range(13):
        days_125.add(d_125)
        d_125 += 7
    return days_125

days_cols = ['day',  'n']
days_data = np.zeros(shape=(100, len(days_cols)))
days_data[0:100, 0] = list(i for i in range(100))
days = pd.DataFrame(days_data, columns=days_cols)

fpath = 'family_data.csv'
data = pd.read_csv(fpath)
#for n in range(10):data['choice_'+str(n)] = data['choice_'+str(n)]-1
#for n in range(10):data['choice_'+str(n)] = data['choice_'+str(n)]+2
days_125 = set_days_125()

data['assigned_day'] = -1
d = 0
for i in range(5000):
    if data.loc[i, 'assigned_day'] == -1:
        n = data.loc[i, 'n_people']
        if days.loc[d, 'n'] + n < 126 and days.loc[d, 'n'] + n != 124:
            days.loc[d, 'n'] += n
            data.loc[i, 'assigned_day'] = d
        if days.loc[d, 'n'] == 125:
            d += 1
            if d == 100:
                break

days_125 = set_days_125()
d = 0
for i in range(5000):
    if data.loc[i, 'assigned_day'] == -1:
        n = data.loc[i, 'n_people']
        if days.loc[d, 'n'] + n < 301:
            days.loc[d, 'n'] += n
            data.loc[i, 'assigned_day'] = d
        d += 1
        if d in days_125:
            d += 1
        if d == 100:
            d = 0

days.to_csv('days_.csv')
data.to_csv('data.csv')