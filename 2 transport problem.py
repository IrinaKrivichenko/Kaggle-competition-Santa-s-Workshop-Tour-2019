import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")
'''
n_days = 5
n_of_choices = 3
def occupancy_limit_function(day):
    return 6
'''
n_days = 6
n_of_choices = 3
def occupancy_limit_function(day):
    return min(round(300 - (300-125)/100 * (day-1)), 300)

def cost_of_lack(day):
    global days
    lim = occupancy_limit_function(day)
    return abs(max(lim - days.loc[day, 'n_people_want'] , 0))*100
    if day == 83 :return 5000
    else:
    return 0

def argmin(matr):
    global n_days
    min_els = np.ndarray.argmin(matr, axis=1)
    min_el = np.ndarray.argmin(matr[ range(n_days), min_els])
    #print('(' + str(min_el) + '; ' + str(min_els[min_el]) + '):' + str(tr_matr[2, min_el, min_els[min_el]]))
    return [min_el, min_els[min_el]]



days_cols = ['day', 'n_fam_want', 'n_people_want', 'tax', 'n', 'lim_func', 'cost_of_lack']
days_data = np.zeros(shape=(n_days, len(days_cols) ))
days_data[0:n_days, 0] = list(i for i in range(n_days))
days = pd.DataFrame(days_data, columns=days_cols)
del days_cols
days.day = days.day # + 1


#fpath = 'test_fam_data.csv'
#fpath = '_fam_data_out_of_bounds.csv'
fpath = 's__fam_data_129123.csv'
data = pd.read_csv(fpath)
data.assigned_day = data.assigned_day - 1
n_famalies = len(data)
#for n in range(n_of_choices):data['choice_'+str(n)] = data['choice_'+str(n)]-1

#fpath = 'sample_submission.csv'
#ss = pd.read_csv(fpath)

dict = {}
tr_matr = np.zeros(shape=(3, n_days, n_famalies+1)).astype(int)
for i in range(n_famalies):
    n = data.loc[i, 'n_people']
    d = data.loc[i, 'choice_0']
    #days.loc[d, 'n_fam_want'] = days.loc[d, 'n_fam_want'] + 1
    #days.loc[d, 'n_people_want'] = days.loc[d, 'n_people_want'] + n
    dict[data.loc[i, 'assigned_day'], i] = n
    tr_matr[0, data.loc[i, 'assigned_day'].astype(int), i] = n
    for d in range(n_days):
        tr_matr[1, d, i] = 500 + 36 * n + 398 * n
    tr_matr[1, data.loc[i, 'choice_0'], i] = 0
    tr_matr[1, data.loc[i, 'choice_1'], i] = 50
    tr_matr[1, data.loc[i, 'choice_2'], i] = 50 + 9 * n
    '''
    tr_matr[1, data.loc[i, 'choice_3'], i] = 100 + 9 * n
    tr_matr[1, data.loc[i, 'choice_4'], i] = 200 + 9 * n
    tr_matr[1, data.loc[i, 'choice_5'], i] = 200 + 18 * n
    tr_matr[1, data.loc[i, 'choice_6'], i] = 300 + 18 * n
    tr_matr[1, data.loc[i, 'choice_7'], i] = 300 + 36 * n
    tr_matr[1, data.loc[i, 'choice_8'], i] = 400 + 36 * n
    tr_matr[1, data.loc[i, 'choice_9'], i] = 500 + 36 * n + 199 * n
'''
data.assigned_day = data.assigned_day.astype(int) + 1
data['assigned_day2'] = 0

for d in range(n_days):
    lim = occupancy_limit_function(d)
    days.loc[d, 'lim_func'] = lim
    days.loc[d, 'cost_of_lack'] = cost_of_lack(d)
    lack_of_people = lim - sum(tr_matr[0, d, :])
    tr_matr[0, d, n_famalies] = lack_of_people
    dict[d, n_famalies] = lack_of_people
del lim
tr_matr[1, :, -1] = days['cost_of_lack']
t = 0
min = -1
min_total = 30000000
set_of_families_included_in_the_basis = set()
while min<0:
    total = 0
    for i in range(n_famalies):
        for d in range(n_days):
            if ( tr_matr[0, d, i]>0):
                total += tr_matr[1, d, i]

    if min_total >= total:
        min_total = total
        reserved_tr_matr = tr_matr[0, :, :]
        reserved_dict = dict
        print('total = '+str(total))
        for i in range(n_famalies):
            for d in range(n_days):
                if (tr_matr[0, d, i] > 0):
                    data.loc[i, 'assigned_day2'] = d + 1
        data.to_csv('' + fpath)
    else:
        tr_matr[0, :, :] = reserved_tr_matr
        dict = reserved_dict

    v = np.full(n_days, np.nan)
    v[0]=0
    u = np.full(n_famalies+1, np.nan)

    while any(np.isnan(v)) or any(np.isnan(u)):
        for key in dict.keys():
            if np.isnan(v[key[0]]):
                if not np.isnan(u[key[1]]):
                    v[key[0]] = tr_matr[1, key[0], key[1]] - u[key[1]]
            if np.isnan(u[key[1]]):
                if not np.isnan(v[key[0]]):
                    u[key[1]] = tr_matr[1, key[0], key[1]] - v[key[0]]
    print('v =' )
    print(v)
    print('u =')
    print(u)

    for i in range(n_famalies):
        for d in range(n_days):
            tr_matr[2, d, i] = tr_matr[1, d, i] - (v[d] + u[i])
    del u
    del v

    pd.DataFrame(tr_matr[0, :, :]).to_csv(f'{t}tr_matr0.csv')
    pd.DataFrame(tr_matr[1, :, :]).to_csv(f'{t}tr_matr1.csv')
    pd.DataFrame(tr_matr[2, :, :]).to_csv(f'{t}tr_matr2.csv')
    t += 1
    ind_min =argmin(tr_matr[2, :, :])
    while ind_min[1] in set_of_families_included_in_the_basis:
        tr_matr[2, ind_min[0], ind_min[1]] = 0
        ind_min = argmin(tr_matr[2, :, :])
    set_of_families_included_in_the_basis.add(ind_min[1])
    min = tr_matr[2, ind_min[0], ind_min[1]]
    print('min = '+ str(min)+ ' on ('+ str(ind_min[0])+','+ str(ind_min[1])+')')
    if min<0:
        circle = []
        max_el = np.argmax(tr_matr[0, :, ind_min[1]])
        circle.append([max_el, ind_min[1]])
        circle.append([ind_min[0], ind_min[1]])
        #print(max_el)
        n = data.loc[ind_min[1], 'n_people']
        if n <= tr_matr[0, ind_min[0], n_famalies]:
            circle.append([ind_min[0], n_famalies])
            circle.append([max_el, n_famalies])
            print('swap family day through dummy column')
        else:
            ind_cand = -1
            for i in range(n_famalies):
                if tr_matr[0, ind_min[0], i] == n:
                    if tr_matr[1, max_el, i] < tr_matr[1, ind_min[0], i]:
                        print('change 2 famalies for days')
                        ind_cand = i
                        break
            if ind_cand != -1:
                circle.append([ind_min[0], ind_cand])
                circle.append([max_el, ind_cand])
            else:
                not_accomplished_days = set()
                for d in range(n_days):
                    if n <= tr_matr[0, d, n_famalies]:
                        not_accomplished_days.add(d)

                break_flag = False
                choice_diff = 11
                for i in range(n_famalies):
                    choice_from = 'lower'
                    if tr_matr[0, ind_min[0], i] == n:
                        for c in range(n_of_choices):
                            cur_choice = data.loc[i, 'choice_'+str(c)]
                            if cur_choice == ind_min[0]:
                                choice_from = c
                            if cur_choice in not_accomplished_days:
                                if choice_from != 'lower':
                                    if choice_diff > (c - choice_from):
                                        ind_cand = i
                                        day_by_choice = cur_choice
                                        choice_diff = choice_from - c
                                        print('choice_diff = '+str(choice_diff))
                                        print('family ' + str(i) + " from choice_" + str(
                                            choice_from) + ' to choice_' + str(c))
                                        continue
                                else:
                                    print('family '+str(i)+" from choice_"+str(choice_from)+' to choice_'+str(c))
                                    circle.append([ind_min[0], i])
                                    circle.append([cur_choice, i])
                                    circle.append([cur_choice, n_famalies])
                                    circle.append([max_el, n_famalies])
                                    ind_cand = 1
                                    break_flag = True
                                    break
                    if break_flag:
                        break
                if ind_cand == -1: continue
                if not break_flag:
                    #print('final choice_diff = '+str(choice_diff))
                    circle.append([ind_min[0], ind_cand])
                    circle.append([day_by_choice, ind_cand])
                    circle.append([day_by_choice, n_famalies])
                    circle.append([max_el, n_famalies])
                #print('change 2 famalies for days through dummy column')
        print(circle)
        while len(circle) > 0:
            a = circle.pop()
            tr_matr[0, a[0], a[1]] += n
            dict[a[0], a[1]] = tr_matr[0, a[0], a[1]]
            a = circle.pop()
            tr_matr[0, a[0], a[1]] -= n
            if a[1] == n_famalies:
                dict[a[0], a[1]] = tr_matr[0, a[0], a[1]]
            else:
                del dict[a[0], a[1]]





