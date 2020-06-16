import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")

days_cols = ['day', 'n',  'tax', 'gifts']
days_data = np.zeros(shape=(100, len(days_cols)))
days_data[0:100, 0] = list(i for i in range(100))
days = pd.DataFrame(days_data, columns=days_cols)
days.day = days.day+1

#fpath = 'fam_data_sin 50 122868.csv'
fpath = 'fam_data_1.csv'
fpath = 'fam_data_discrete.csv'
fpath = 'fam_data_253758.98569519774.csv'
fpath = 'fam_data_253758.csv'
#fpath = 'fam_data_0_discrete.csv'
#fpath = 'fam_data_0_liner_upto_309.csv'

'''
fpath = '_fam_data_121017.csv'
fpath = '_fam_data_122812.csv'
fpath = '_fam_data_123341.csv'
fpath = '_fam_data_123896.csv'
fpath = 'new_fam_data_.csv'
fpath = 'gen1_1.csv'
'''
#fpath = '_fam_data_out_of_bounds.csv'
data = pd.read_csv(fpath)

#fpath = 'sample_submission.csv'
#ss = pd.read_csv(fpath)

def occupancy_limit_function( day):
    # return min(round(350 - (300-125)/100 * (day)), 300)
    if day < 4: return 300
    line = 309 - (300 - 125) / (100 - 11) * (day)
    func = np.sin(day * 5.65 / (2 * np.pi) + 5) * 55 + line
    # func = line
    if func < 125: return 125
    if func > 300: return 300
    if func < line: return line
    return round(func)

gifts = 0
gifts_mondays = 0
gifts_sundays = 0
gifts_saturdays = 0
gifts_fridays = 0
gifts_thursdays = 0
gifts_wednesdays = 0
gifts_tuesdays = 0
data['choice_num'] = 0
data['gift'] = 0

days_125 = set()
d_125 = 8
for _ in range(13):
    days_125.add(d_125)
    d_125 += 7
print(days_125)

for i in range(5000):
    gift = -1
    n = data.loc[i, 'n_people']
    d = data.loc[i, 'assigned_day']-1
    if d != -1:
        days.loc[d, 'n'] += n

        for j in range(10):
            if d == data.loc[i, 'choice_'+str(j)]:
                data.loc[i, 'choice_num'] = j+1
                if j == 0:
                    gift = 0
                elif j == 1:
                    gift = 50
                elif j == 2:
                    gift = 50 + 9 * n
                elif j == 3:
                    gift = 100 + 9 * n
                elif j == 4:
                    gift = 200 + 9 * n
                elif j == 5:
                    gift = 200 + 18 * n
                elif j == 6:
                    gift = 300 + 18 * n
                elif j == 7:
                    gift = 300 + 36 * n
                elif j == 8:
                    gift = 400 + 36 * n
                elif j == 9:
                    gift = 500 + 36 * n + 199 * n
                break
        if gift == -1 :
            gift = 500 + 36 * n + 398 * n
        days.loc[d, 'gifts'] += gift
        if d in days_125:
            gifts_mondays = gifts_mondays + gift
        elif d-1 in days_125:
            gifts_sundays = gifts_sundays + gift
        elif d - 2 in days_125:
            gifts_saturdays = gifts_saturdays + gift
        elif d - 3 in days_125:
            gifts_fridays = gifts_fridays + gift
        elif d - 4 in days_125:
            gifts_thursdays = gifts_thursdays + gift
        elif d - 5 in days_125:
            gifts_wednesdays = gifts_wednesdays + gift
        elif d - 6 in days_125:
            gifts_tuesdays = gifts_tuesdays + gift
        data.loc[i, 'gift'] = gift
        gifts = gifts + gift
                # Calculate the gift for not getting top preference
#data.to_csv(fpath)
accounting_penalty = 0
Ndprev = days.loc[99, 'n']
for d in reversed(range(100)):
    Nd = days.loc[d, 'n']
    day_tax = (Nd - 125)/400  *  Nd ** (0.5+ abs(Nd-Ndprev)/50)
    if Nd < 125 or Nd > 300: day_tax = 100000000
    days.loc[d, 'tax'] = day_tax
    lim = occupancy_limit_function(d)
    accounting_penalty = accounting_penalty + day_tax
    Ndprev = Nd
print(days)



print("accounting_penalty = ", accounting_penalty)
print("gifts = ", gifts)
print("gifts_mondays = ", gifts_mondays)
print("gifts_sundays = ", gifts_sundays)
print("gifts_saturdays = ", gifts_saturdays)
print("gifts_fridays = ", gifts_fridays)
print("gifts_thursdays = ", gifts_thursdays)
print("gifts_wednesdays = ", gifts_wednesdays)
print("gifts_tuesdays = ", gifts_tuesdays)
print("total outgo = ", gifts + accounting_penalty)

days.to_csv('days_'+fpath)