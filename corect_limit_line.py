import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")
def count_gift(family_id, d, for_counting_results):
    gift = -1
    n = data.loc[family_id, 'n_people']
    for j in range(10):
        if d == data.loc[family_id,'choice_' + str(j)]:
            if for_counting_results:
                data.loc[family_id,'choice_num'] = j
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
    if gift == -1:
        gift = 500 + 36 * n + 398 * n
    data.loc[family_id, 'gift'] = gift
    return gift

def count_d_tax(day, n, for_counting_results):
    if day == 99:
        Ndprev = days.loc[99, 'n']
    else:
        Ndprev = days.loc[day+1, 'n']
    Nd = days.loc[day, 'n']
    day_tax = ((Nd+n) - 125) / 400 * (Nd+n) ** (0.5 + abs((Nd+n) - Ndprev) / 50)
    if Nd < 125 or Nd > 300:
        print(f"!!!!!   out of bounds   !!!!! day {day} Nd = {Nd}")
        exit()
    if for_counting_results or day == 0:
        return day_tax
    Ndprev = (Nd + n)
    Nd = days.loc[day-1, 'n']
    day_tax += (Nd - 125) / 400 * Nd ** (0.5 + abs(Nd - Ndprev) / 50)
    return day_tax

def regard_a_family(family_id):
    global data
    global days
    global improvement
    days_125 = set()
    d_125 = 8
    for _ in range(13):
        days_125.add(d_125)
        d_125 += 7
    n = data.loc[family_id,'n_people']
    ad = data.loc[family_id,'assigned_day']-1
    cn = data.loc[family_id,'choice_num']
    if (days.loc[ad, 'n'] - n)>=125 :
        g = data.loc[family_id,'gift']
        current_tax_ad = count_d_tax(ad, 0, False)
        expected_tax_ad = count_d_tax(ad, -n, False)
        for c in range(10):
            d = data.loc[family_id,'choice_' + str(c)]
            if (days.loc[d, 'n'] + n) <= 300 and d not in days_125:
                current_tax_d = count_d_tax(d, 0, False)
                expected_tax_d = count_d_tax(d, +n, False)
                tax_diff = (current_tax_ad + current_tax_d) - (expected_tax_ad + expected_tax_d)
                gift_diff = g - count_gift(family_id, d, False)
                if(-tax_diff < gift_diff):
                    data.loc[family_id, 'assigned_day'] = d+1
                    data.loc[family_id, 'choice_num'] = c
                    days.loc[ad, 'n'] -= n
                    days.loc[d, 'n'] += n
                    improvement = improvement + tax_diff + gift_diff
                    break


days_cols = ['day', 'n']
days_data = np.zeros(shape=(100, len(days_cols)))
days_data[0:100, 0] = list(i for i in range(100))
days = pd.DataFrame(days_data, columns=days_cols)
days.day = days.day-1

#fpath = 'fam_data_0_liner_upto_309.csv'
fpath = 'fam_data_0_liner.csv'
#fpath = 'fam_data_253758.98569519774.csv'
#fpath = 'fam_data_8711127.csv'
#fpath = 'fam_data_0_sin.csv'
#fpath = 'fam_data_0_discrete.csv'
#fpath = 'fam_data_0_sorted.csv'
data = pd.read_csv(fpath)
#for n in range(10):data['choice_'+str(n)] = data['choice_'+str(n)]+2
#fpath = 'sample_submission.csv'
#ss = pd.read_csv(fpath)
total_outgo = 0
def count_results():
    global data
    global days
    global total_outgo
    consolation_gifts = 0
    data['choice_num'] = 0
    data['gift'] = 0
    days['n']=0
    for i in range(5000):
        gift = -1
        n = data.loc[i, 'n_people']
        d = data.loc[i, 'assigned_day']-1
        days.loc[d, 'n'] += n
        gift = count_gift(data.loc[i, 'family_id'], d , True)
        data.loc[i, 'gift'] = gift
        consolation_gifts = consolation_gifts + gift
                # Calculate the gift for not getting top preference
    #data.to_csv(fpath)
    accounting_penalty = 0
    Ndprev = days.loc[99, 'n']
    for d in reversed(range(100)):
        Nd = days.loc[d, 'n']
        day_tax = (Nd - 125)/400  *  Nd ** (0.5+ abs(Nd-Ndprev)/50)
        if Nd < 125 or Nd > 300: day_tax = 100000000
        days.loc[d, 'tax'] = day_tax
        accounting_penalty = accounting_penalty + day_tax
        Ndprev = Nd
    print("accounting_penalty = ", accounting_penalty)
    print("consolation_gifts = ", consolation_gifts)
    total_outgo = consolation_gifts + accounting_penalty
    print("total outgo = ", total_outgo)

count_results()
improvement = 0
data = data.sort_values(by='gift', ascending=False)
#data.to_csv('fam_data_0_sorted.csv')
print(data.head())
for i in range(5000):
    regard_a_family(data.iloc[i]['family_id'])

data = data.sort_values(by='family_id')
print(total_outgo-improvement)
count_results()

data.to_csv(fpath, index=False)




print('days_'+fpath)
days.to_csv('days_'+fpath)