import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")
'''
n_days = 6
n_of_choices = 3
def occupancy_limit_function(day):
    return 15
'''
n_days = 100
n_of_choices = 10

def argmin(matr):
    global n_days
    min_els = np.ndarray.argmin(matr, axis=1)
    min_el = np.ndarray.argmin(matr[ range(n_days), min_els])
    #print('(' + str(min_el) + '; ' + str(min_els[min_el]) + '):' + str(tr_matr[2, min_el, min_els[min_el]]))
    return [min_el, min_els[min_el]]

def fill_in_u_and_v(it_is_u, key):
    global tr_matr
    global v
    global u
    global dict_d_as_key
    global dict_f_as_key
    if it_is_u:
        if not np.isnan(u[key]):
            for i in dict_f_as_key[key]:
                if np.isnan(v[i]):
                    v[i] = tr_matr[1, i, key] - u[key]
                    fill_in_u_and_v(False, i)
        else:
            return
    else:
        if not np.isnan(v[key]):
            for i in dict_d_as_key[key]:
                if np.isnan(u[i]):
                    u[i] = tr_matr[1, key, i] - v[key]
                    fill_in_u_and_v(True, i)
        else:
            return

def find_var_to_except_from_basis(d , n, min):
    global tr_matr
    global u
    global dict_d_as_key
    min_i = -1
    for i in dict_d_as_key[d]:
        if tr_matr[0, d, i] == n:
            cur_min = np.min(tr_matr[2, :, i])
            if cur_min < min:
                min = cur_min
                min_i = i
    return min_i

def pop_2_from_circle():
    global circle
    global dict_d_as_key
    global tr_matr
    global amount_to_change
    a = circle.pop()
    if a[1] in dict_d_as_key[a[0]]:
        added = False
    else:
        added = True
    tr_matr[0, a[0], a[1]] += amount_to_change
    dict_d_as_key[a[0]].add(a[1])
    dict_f_as_key[a[1]].add(a[0])
    a = circle.pop()
    tr_matr[0, a[0], a[1]] -= amount_to_change
    if tr_matr[0, a[0], a[1]] < 0:
        exit()
    if added:
        print('  going to remove' + str(a) + '    in matrix ' + str(tr_matr[0, a[0], a[1]]))
        dict_d_as_key[a[0]].remove(a[1])
        dict_f_as_key[a[1]].remove(a[0])
    return a[0]

days_cols = ['day', 'n_fam_want', 'n_people_want', 'tax', 'n', 'lim_func', 'cost_of_lack']
days_data = np.zeros(shape=(n_days, len(days_cols) ))
days_data[0:n_days, 0] = list(i for i in range(n_days))
days = pd.DataFrame(days_data, columns=days_cols)
del days_cols
days.day = days.day # + 1


fpath = 'fam_data_0_liner_upto_309.csv'
fpath = 'fam_data_0_liner.csv'
#fpath = 'fam_data_0_sin.csv'
#fpath = 'fam_data_0_sorted.csv'
#fpath = 'fam_data_0_discrete.csv'
#fpath = 'fam_data_1.csv'
#fpath = 'fam_data_sin 50 122868.csv'
#fpath = '_fam_data_121017.csv'
#fpath = '_out_of_bounds.csv'
data = pd.read_csv(fpath)
data.assigned_day = data.assigned_day - 1
n_famalies = len(data)
#for n in range(n_of_choices):data['choice_'+str(n)] = data['choice_'+str(n)]-1

#fpath = 'sample_submission.csv'
#ss = pd.read_csv(fpath)

dict_f_as_key = {}
for f in range(n_famalies):
    dict_f_as_key[f] = set()
dict_d_as_key = {}
for d in range(n_days):
    dict_d_as_key[d] = set()
print('')


tr_matr = np.zeros(shape=(4, n_days, n_famalies)).astype(int)
for i in range(n_famalies):
    n = data.loc[i, 'n_people']
    d = data.loc[i, 'choice_0']
    #days.loc[d, 'n_fam_want'] = days.loc[d, 'n_fam_want'] + 1
    #days.loc[d, 'n_people_want'] = days.loc[d, 'n_people_want'] + n
    dict_d_as_key[data.loc[i, 'assigned_day']].add(i)
    dict_f_as_key[i].add(data.loc[i, 'assigned_day'])
    tr_matr[0, data.loc[i, 'assigned_day'].astype(int), i] = n
    for d in range(n_days):
        tr_matr[1, d, i] = 500 + 36 * n + 398 * n
    tr_matr[1, data.loc[i, 'choice_0'], i] = 0
    tr_matr[1, data.loc[i, 'choice_1'], i] = 50
    tr_matr[1, data.loc[i, 'choice_2'], i] = 50 + 9 * n
    #'''
    tr_matr[1, data.loc[i, 'choice_3'], i] = 100 + 9 * n
    tr_matr[1, data.loc[i, 'choice_4'], i] = 200 + 9 * n
    tr_matr[1, data.loc[i, 'choice_5'], i] = 200 + 18 * n
    tr_matr[1, data.loc[i, 'choice_6'], i] = 300 + 18 * n
    tr_matr[1, data.loc[i, 'choice_7'], i] = 300 + 36 * n
    tr_matr[1, data.loc[i, 'choice_8'], i] = 400 + 36 * n
    tr_matr[1, data.loc[i, 'choice_9'], i] = 500 + 36 * n + 199 * n
#'''
#data.assigned_day = data.assigned_day.astype(int) + 1
#data['assigned_day2'] = 0
t = 0
min = -1
min_total = 30000000
prev_ind_min = [0, 0]
set_of_families_included_in_the_basis = []
for t in range(90):
    total = 0
    for i in range(n_famalies):
        for d in range(n_days):
            if ( tr_matr[0, d, i]>0):
                total += tr_matr[1, d, i]
    print('t = ' + str(t))
    if min_total >= total:
        not_circle = False
        min_total = total
        print('total = '+str(total))
        for i in range(n_famalies):
            for d in range(n_days):
                if (tr_matr[0, d, i] > 0):
                    data.loc[i, 'assigned_day'] = d + 1
        data.to_csv('' + fpath, index=False)
        data.to_csv('_Copy' + fpath , index=False)



    v = np.full(n_days, np.nan)
    v[0]=0
    u = np.full(n_famalies, np.nan)

    fill_in_u_and_v(False, 0)
    while any(np.isnan(v)) or any(np.isnan(u)):
        v_index = np.where(np.isnan(v))[0].astype(int)[0]
        v[v_index] = 0
        fill_in_u_and_v(False, v_index)
    #print('v =' )
    #print(v)
    #print('u =')
    #print(u)

    for i in range(n_famalies):
        for d in range(n_days):
            tr_matr[2, d, i] = tr_matr[1, d, i] - (v[d] + u[i])

    '''
    pd.DataFrame(tr_matr[0, :, :]).to_csv(f'{t}tr_базис.csv')
    pd.DataFrame(tr_matr[1, :, :]).to_csv(f'{t}tr_стоймости.csv')
    pd.DataFrame(tr_matr[2, :, :]).to_csv(f'{t}tr_разность.csv')
    #'''
    ind_min =argmin(tr_matr[2, :, :])
    while ind_min[0] == 0:
        tr_matr[2, ind_min[0], ind_min[1]] = 0
        ind_min = argmin(tr_matr[2, :, :])
    if ind_min == prev_ind_min:
        ind_min_count += 1
        if ind_min_count > 1:
            dict_d_as_key[ind_min[0]].add(ind_min[1])
            dict_f_as_key[ind_min[1]].add(ind_min[0])
    else:
        prev_ind_min = ind_min
        ind_min_count = 0
    '''
    while ind_min[0] ==0 :#in set_of_families_included_in_the_basis:
        tr_matr[2, ind_min[0], ind_min[1]] = 0
        ind_min = argmin(tr_matr[2, :, :])
    #set_of_families_included_in_the_basis.add(ind_min[1])
    #'''
    min = tr_matr[2, ind_min[0], ind_min[1]]
    print('min = '+ str(min)+ ' on ('+ str(ind_min[0])+','+ str(ind_min[1])+')')
    next_min = False
    short_circle = False
    while min<0:
        circle = []
        ind_max = np.argmax(tr_matr[0, :, ind_min[1]])
        n = tr_matr[0, ind_max, ind_min[1]]
        circle.append([ind_max, ind_min[1]]) # ---
        circle.append([ind_min[0], ind_min[1]])  # +++
        d_plus = ind_min[0]
        days_already_in_circle = set()
        while  d_plus != ind_max:
            #if t == 16:
            #    print('t == 9')
            d_minus = d_plus
            f = find_var_to_except_from_basis(d_minus, n, -min)
            if f == -1:     # no famaly in that day
                #print('f == -1')

                if len(circle) == 2:
                    next_min = True
                    break
                a = circle.pop() #go back to the previous family and change day by another choice
                tr_matr[2, a[0], a[1]] = 10000
                f = a[1]
            else:
                circle.append([d_minus, f])  # ---
            d_plus = np.argmin(tr_matr[2, :, f])
            ttl = 2
            while d_plus == d_minus :# если
                #print('d_plus == d_minus')
                tr_matr[2, d_plus, f] = 10000
                d_plus = np.argmin(tr_matr[2, :, f])
                if ttl == 0:
                    print('ttl == 0')
                    exit()
                ttl -= 1
            else:
                if tr_matr[2, d_plus, f] >= -min or d_plus in days_already_in_circle :
                    next_min = True
                    break
                circle.append([d_plus, f])  # +++
                days_already_in_circle.add(d_plus)
        if next_min :
            tr_matr[2, ind_min[0], ind_min[1]] = 0
            ind_min = argmin(tr_matr[2, :, :])
            min = tr_matr[2, ind_min[0], ind_min[1]]
            next_min = False
            continue
        amount_to_change = n
        print(circle)
        while len(circle) > 0:
            pop_2_from_circle()
        min = 0
        #ammount_in_dict = 0
        #for key in dict_d_as_key:
        #    ammount_in_dict += len(dict_d_as_key[key])
        #print('len dict_d_as_key =' +str(ammount_in_dict))




