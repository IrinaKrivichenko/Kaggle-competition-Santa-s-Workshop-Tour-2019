# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")
# выводим список файлов из текущей директории
#for dirname, _, filenames in os.walk('.'):
#    for filename in filenames:  print(os.path.join(dirname, filename))

fpath = 'new_fam_data_.csv'
data = pd.read_csv(fpath)
n_fam = 5000
n_choice = 10
#fpath = 'sample_submission.csv'
#submission = pd.read_csv(fpath)
print('data was readed')

# data.head()
data['assigned_day'] = -1

# for col in data.columns: print(col)
n_days = 100
days_cols = ['day', 'current_choice', 'current_n', 'n', 'tax']
days_data = np.zeros(shape=(n_days, len(days_cols)))
days_data[0:n_days, 0] = list(i for i in range(n_days))
days = pd.DataFrame(days_data, columns=days_cols)
days.day = days.day + 1


def look_over_choice(days, choice_col_num):
    global  data
    days = days.sort_values(by=['day'])
    days.current_choice = 0
    days.current_n = 0
    for i in range(n_fam):
        if data['assigned_day'][i] == -1:
            d = data.loc[i, 'choice_'+str(choice_col_num)]
            print('d = ' + str(d))
            print('i = ' + str(i))
            days.loc[d , 'current_n'] = days.loc[d , 'current_n'] + data.loc[i, 'n_people']
    return days#.sort_values(by=['current_n'])

MAX_OCCUPANCY = '1st_liner_func'
def occupancy_limit_function(day):
    #return 300 - (300-125)/100 * (day-1)
    lim = [300, 300, 300, 300, 300, 256, 213, 169, 125, 300,
           300, 300, 256, 213, 169, 125, 300, 300, 300, 256,
           213, 169, 125, 300, 300, 300, 256, 213, 169, 125,
           300, 283, 265, 230, 195, 160, 125, 300, 283, 267,
           233, 200, 163, 125, 300, 265, 230, 195, 148, 125,
           280, 241, 203, 164, 125, 125, 125, 125, 270, 237,
           201, 155, 125, 125, 125, 253, 222, 184, 131, 125,
           125, 125, 250, 219, 188, 156, 125, 125, 125, 235,
           208, 180, 153, 125, 125, 125, 235, 217, 179, 145,
           125, 125, 125, 220, 205, 179, 145, 125, 125, 125]
    return lim[day.astype(int)]

def set_assignment(days, choice_col_num):
    if choice_col_num >1: preferable_choice = False
    else: preferable_choice = True
    number_of_assigned_families = 0
    number_of_assigned_families2 = 0
    for d in range(n_days):
        cur_n = days.iloc[d, 2]
        if cur_n > 0:
            #
           # if cur_n > 300:     break
            day = days.iloc[d, 0]
            n = days.iloc[d, 3]
            if preferable_choice or n > 0:
                l = occupancy_limit_function(day)
                print(f'day {day} lim {l}')
                if (cur_n + days.iloc[d, 3]) <= l:
                    number_of_assigned_families2 = number_of_assigned_families2 + days.iloc[d, 1]
                    for i in range(n_fam):
                        if data['assigned_day'][i] == -1 and data.loc[i, 'choice_'+str(choice_col_num)] == day:
                            # days['n'][d] = days['n'][d]+data['n_people'][i]
                            days.iloc[d, 3] = days.iloc[d, 3] + data.loc[i, 'n_people']
                            # data['assigned_day'][i] = c+1
                            data.loc[i, 'assigned_day'] = day
                            number_of_assigned_families = number_of_assigned_families + 1
    return  number_of_assigned_families

number_of_families = 0

def main_assignment_func(first_time):
    global number_of_families
    global days
    cur_number_of_families = 0
    max_choice_to_look = 1
    while number_of_families < n_fam  and max_choice_to_look < n_choice+1 :
        for i in range(max_choice_to_look):
            if first_time:
                days = look_over_choice(days, i)  # 'choice_0'
                cur_number_of_families = set_assignment(days, i)
                number_of_families = number_of_families + cur_number_of_families
            else:
                first_time = True
        if cur_number_of_families == 0:
            max_choice_to_look = max_choice_to_look + 1
        print("number_of_families = " , number_of_families, "  choice_", max_choice_to_look)
    days = days.sort_values(by=['n'])
    # чтобы выводился весь датафрейм
    #pd.set_option('display.max_rows', None)
    print(days.head())

print('line 103')
main_assignment_func(True)
print('line 107')
DOWN_LIMIT=55
MIN_OCCUPANCY = 125
days = days.sort_values(by=['n'])

non_popular_days = set()
for d in range(n_days):
    if days.iloc[d, 3] <= occupancy_limit_function(days.iloc[d, 0])-DOWN_LIMIT :
        non_popular_days.add(days.iloc[d, 0])
print(sorted(non_popular_days))
non_popular_days.remove(1)

preferable_days = set()
days = days.sort_values(by=['day'])
if number_of_families < n_fam :
    for i in range(n_fam):
        if data.loc[i, 'assigned_day'] == 0:
            preferable_days.add(data.loc[i, 'choice_1'])


print(sorted(non_popular_days))
print(number_of_families)

def re_assignment(days, choice_col_num):
    days = days.sort_values(by=['day'])
    for i in range(n_fam):
        if len(non_popular_days) == 0:
            break
        if data.loc[i, 'assigned_day'] not in non_popular_days:
            if data.iloc[i, choice_col_num] in non_popular_days:
                day_to_change_from = data.loc[i, 'assigned_day']
                day_to_change_to = data.iloc[i, choice_col_num]
                n_people_in_family = data.loc[i, 'n_people']
                n_day_to_change_from = days.loc[day_to_change_from - 1, 'n']
                if n_day_to_change_from - n_people_in_family >= MIN_OCCUPANCY:
                    days.loc[day_to_change_from - 1, 'n'] = n_day_to_change_from - n_people_in_family
                    days.loc[day_to_change_to - 1, 'n'] = days.loc[day_to_change_to - 1, 'n'] + n_people_in_family
                    if days.loc[day_to_change_to - 1, 'n'] >= MIN_OCCUPANCY:
                        non_popular_days.remove(day_to_change_to)
                    data.loc[i, 'assigned_day'] = day_to_change_to
    print('non_popular_days after reassignment by choice_', choice_col_num-1, ' = ', len(non_popular_days))
    return days
print('line 147')
days = re_assignment(days, 2)#'choice_1')
days = re_assignment(days, 3) #'choice_2')
days = re_assignment(days, 4) #'choice_3')
days = re_assignment(days, 5) #'choice_4')
days = re_assignment(days, 6) #'choice_5')
days = re_assignment(days, 7) #'choice_6')
days = re_assignment(days, 8) #'choice_7')
days = re_assignment(days, 9) #'choice_8')
days = re_assignment(days, 10) #'choice_9')


for i in range(n_fam):
    d = data.loc[i, 'assigned_day']
    if  d in preferable_days and d != data.loc[i, 'choice_0'] : #and d != data.loc[i, 'choice_1'] and d != data.loc[i, 'choice_2']:
        days.loc[d, "n"] = days.loc[d, "n"] - data.loc[i, 'n_people']
        data.loc[i, 'assigned_day'] = 0
        number_of_families = number_of_families - 1

print("number_of_families after re_assignment ", number_of_families)

main_assignment_func(False)

days = days.sort_values(by=['day'])
print(days.head())

days = days.sort_values(by=['day'])
days['tax'] = 0
accounting_penalty = 0
Ndprev = days.iloc[n_days-1, 3]
for d in reversed(range(n_days)):
    Nd = days.iloc[d, 3]

#    day_tax2_power =  0.5+ abs(Nd-Ndprev)/50
#    day_tax2 =  Nd ^ day_tax2_power
#    day_tax = day_tax1 * day_tax2

    day_tax = (Nd - 125)/400  *  Nd ** (0.5+ abs(Nd-Ndprev)/50)
    days.iloc[d, 4] = day_tax
    accounting_penalty = accounting_penalty + day_tax
    Ndprev = Nd

print("MAX_OCCUPANCY = ",MAX_OCCUPANCY, " with DOWN_LIMIT =",DOWN_LIMIT)
print("accounting_penalty = ",accounting_penalty)
# чтобы выводился весь датафрейм
#pd.set_option('display.max_rows', None)
#print(days)
days.to_csv(f'days_DOWN_LIMIT_{DOWN_LIMIT}.csv')

consolation_gifts = 0
data['choice_num'] = 0
data['gift'] = 0
data['consolation_gifts'] = 0

for i in range(n_fam):
    gift = -1
    n = data.loc[i, 'n_people']
    for j in range(n_choice):
        if data.loc[i, 'assigned_day'] == data.loc[i, 'choice_'+str(j)]:
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
    data.loc[i, 'gift'] = gift
    consolation_gifts = consolation_gifts + gift
    data.loc[i, 'consolation_gifts'] = consolation_gifts
            # Calculate the gift for not getting top preference


print("consolation_gifts = ", consolation_gifts)

print("total outgo = ", consolation_gifts + accounting_penalty)
data.to_csv(fpath)

