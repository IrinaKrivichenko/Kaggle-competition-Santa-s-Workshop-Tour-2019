import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")
import random

def occupancy_limit_function(day):
    return min(round(300 - (300-125)/100 * (day)), 300)

days_cols = ['day', 'families', 'n', 'lim_func', 'free']
days_data = np.zeros(shape=(100, len(days_cols)))
days_data[0:100, 0] = list(i for i in range(100))
days = pd.DataFrame(days_data, columns=days_cols)
for d in range(100):
    lim = occupancy_limit_function(d)
    days.loc[d, 'lim_func'] = lim
    days.loc[d, 'free'] = lim

n_of_choices = 10
fpath = 'family_data.csv'
data = pd.read_csv(fpath)
n_famalies = len(data)
for n in range(n_of_choices):data['choice_'+str(n)] = data['choice_'+str(n)]-1
data['assigned_day'] = -1

n_of_population = 2
datas = []
datas_to_remove = []
dayses = []
dayses_to_remove = []
for i in range(n_of_population):
    datas.append(data)
    datas_to_remove.append(data)
    dayses.append(days)
    dayses_to_remove.append(days)
for m in range(n_famalies):
    for i in range(n_of_population):
            j = random.randint(0, len(datas_to_remove[i])-1)
            n = datas_to_remove[i].iloc[j, 11]#'n_people'
            f = datas_to_remove[i].iloc[j, 0 ]#'family_id'
            d_ok = False
            for c in range(10):
                d = datas_to_remove[i].iloc[j, c+1 ]#'choice_' + str(c)]
                if dayses[i].loc[d, 'free'] >= n:
                    d_ok = True
                    break
            while not d_ok:
                dd = random.randint(0, len(dayses_to_remove[i]) - 1)
                d = dayses_to_remove[i].iloc[dd, 0]  # 'day'
                if dayses[i].loc[d, 'free'] >= n:
                    d_ok = True
            dayses[i].loc[d, 'n'] += n
            dayses[i].loc[d, 'families'] += 1
            dayses[i].loc[d, 'free'] -= n
            datas[i].loc[f, 'assigned_day'] = d
            datas_to_remove[i] = datas_to_remove[i].drop(f)
            if dayses[i].loc[d, 'free'] < 2:
                print(str(d) + " on step " + str(m))
                dayses_to_remove[i] = dayses_to_remove[i].drop(d)



for i in range(n_of_population):
    datas[i]['assigned_day'] += 1
    datas[i].to_csv("gen1_"+str(i)+'.csv')
    dayses[i].to_csv("days_gen1_"+str(i)+'.csv')
