import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import time
import threading
from operator import attrgetter

os.chdir("I:/Users/Ira/PycharmProjects/untitled/kaggle/input/")

class Individual:
    def __init__(self, data, generation):
        self.cost_of_lack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 11, 4, 6, 11, 0, 0, 0, 11, 11, 26, 10, 0, 0, 0, 12, 0, 0, 19, 0,
                             0, 0, 11, 22, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 22, 0, 0, 0]
        self.n_days = len(self.cost_of_lack)
        self.data = data.copy()
        self.n_famalies = len(data)
        self.n_of_choices = 10
        self.generation = generation
        self.total_outgo = 'not counted'
        self.days_125 = set()
        d_125 = 8
        for _ in range(13):
            self.days_125.add(d_125)
            d_125 += 7

        if 'assigned_day' in self.data.columns:
            pass
        else:
            self.data['assigned_day'] = -1

    def generate(self):
        # print('going to generate')
        days_cols = ['day', 'families', 'n', 'lim_func', 'free']
        days_data = np.zeros(shape=(100, len(days_cols)))
        days_data[0:100, 0] = list(i for i in range(100))
        days = pd.DataFrame(days_data, columns=days_cols)
        for d in range(self.n_days):
            lim = self.occupancy_limit_function(d)
            days.loc[d, 'lim_func'] = lim
            days.loc[d, 'free'] = lim
        self.data['assigned_day'] = -1
        data_to_remove = self.data.copy()
        days_to_remove = days.copy()
        while len(data_to_remove) > 0:
            j = random.randint(0, len(data_to_remove) - 1)
            n = data_to_remove.iloc[j, 11]  # 'n_people'
            f = data_to_remove.iloc[j, 0]  # 'family_id'
            d_ok = False
            for c in range(self.n_of_choices):
                d = data_to_remove.iloc[j, c + 1]  # 'choice_' + str(c)]
                if days.loc[d, 'free'] >= n:
                    d_ok = True
                    break
            while not d_ok:
                dd = random.randint(0, len(days_to_remove) - 1)
                d = days_to_remove.iloc[dd, 0]  # 'day'
                if days.loc[d, 'free'] >= n:
                    d_ok = True
            days.loc[d, 'n'] += n
            days.loc[d, 'families'] += 1
            days.loc[d, 'free'] -= n
            self.data.loc[f, 'assigned_day'] = d + 1
            data_to_remove = data_to_remove.drop(f)
            if days.loc[d, 'free'] < 2:
                days_to_remove = days_to_remove.drop(d)

    def set_assigned_day(self, f, d):
        self.data.loc[f, 'assigned_day'] = d

    def lead_to_borders(self):
        while np.min(self.data['assigned_day']) < 0:
            self.data['assigned_day'] += 1
        while np.max(self.data['assigned_day']) > 99:
            self.data['assigned_day'] -= 1
        self.tr_matr = np.zeros(shape=(4, self.n_days, self.n_famalies + 1)).astype(int)
        self.dict_f_as_key = {}
        for f in range(self.n_famalies + 1):
            self.dict_f_as_key[f] = set()
        self.dict_d_as_key = {}
        for d in range(self.n_days):
            self.dict_d_as_key[d] = set()
        for i in range(self.n_famalies):
            n = self.data.loc[i, 'n_people']
            d = self.data.loc[i, 'assigned_day'].astype(int)
            self.dict_d_as_key[d].add(i)
            self.dict_f_as_key[i].add(d)
            self.tr_matr[0, d, i] = n
        for d in range(self.n_days):
            lim = self.occupancy_limit_function(d)
            # days.loc[d, 'lim_func'] = lim
            lack_of_people = lim - sum(self.tr_matr[0, d, :])
            self.tr_matr[0, d, self.n_famalies] = lack_of_people
            self.dict_d_as_key[d].add(self.n_famalies)
            self.dict_f_as_key[self.n_famalies].add(d)
        print('going to lead_to_top_border')
        self.lead_to_top_border()
        print('going to lead_to_low_border')
        self.lead_to_low_border()

        for i in range(self.n_famalies):
            for d in range(self.n_days):
                if (self.tr_matr[0, d, i] > 0):
                    self.data.loc[i, 'assigned_day'] = d + 1
        del self.tr_matr
        fpath = 'new_fam_data_.csv'
        #self.data.to_csv('' + fpath)
        self.count()
        self.save(self.total_outgo)

    def occupancy_limit_function(self, day):
        # return min(round(350 - (300-125)/100 * (day)), 300)
        '''
        if day < 4: return 300
        line = min(round(309 - (300 - 125) / 100 * (day)), 300)
        sin = np.sin(day * 5.65 / (2 * np.pi) + 5) * 55 + line
        if sin < 125: return 125
        if sin > 300: return 300
        return round(sin)
        '''
        lim = [300, 300, 300, 271, 242, 213, 184, 155, 125, 300,
                271, 242, 213, 184, 155, 125, 300, 271, 242, 213,
                184, 155, 125, 300, 271, 242, 213, 184, 155, 125,
                300, 271, 242, 213, 184, 155, 125, 300, 271, 242,
                213, 184, 155, 125, 300, 271, 242, 213, 184, 155,
                125, 300, 271, 242, 213, 184, 155, 125, 300, 271,
                242, 213, 184, 155, 125, 300, 271, 242, 213, 184,
                155, 125, 300, 271, 242, 213, 184, 155, 125, 300,
                271, 242, 213, 184, 155, 125, 300, 271, 242, 213,
                184, 155, 125, 300, 271, 242, 213, 184, 155, 125]
        return lim[day]


    def set_cost_value(self):
        for i in range(self.n_famalies):
            n = self.data.loc[i, 'n_people']
            for d in range(self.n_days):
                self.tr_matr[1, d, i] = 500 + 36 * n + 398 * n
            self.tr_matr[1, self.data.loc[i, 'choice_0'], i] = 0
            self.tr_matr[1, self.data.loc[i, 'choice_1'], i] = 50
            self.tr_matr[1, self.data.loc[i, 'choice_2'], i] = 50 + 9 * n
            self.tr_matr[1, self.data.loc[i, 'choice_3'], i] = 100 + 9 * n
            self.tr_matr[1, self.data.loc[i, 'choice_4'], i] = 200 + 9 * n
            self.tr_matr[1, self.data.loc[i, 'choice_5'], i] = 200 + 18 * n
            self.tr_matr[1, self.data.loc[i, 'choice_6'], i] = 300 + 18 * n
            self.tr_matr[1, self.data.loc[i, 'choice_7'], i] = 300 + 36 * n
            self.tr_matr[1, self.data.loc[i, 'choice_8'], i] = 400 + 36 * n
            self.tr_matr[1, self.data.loc[i, 'choice_9'], i] = 500 + 36 * n + 199 * n
        self.tr_matr[1, :, -1] = self.cost_of_lack

    def fill_in_u_and_v(self, it_is_u, key):
        if it_is_u:
            if not np.isnan(self.u[key]):
                for i in self.dict_f_as_key[key]:
                    if np.isnan(self.v[i]):
                        self.v[i] = self.tr_matr[1, i, key] - self.u[key]
                        self.fill_in_u_and_v(False, i)
            else:
                return
        else:
            if not np.isnan(self.v[key]):
                for i in self.dict_d_as_key[key]:
                    if np.isnan(self.u[i]):
                        self.u[i] = self.tr_matr[1, key, i] - self.v[key]
                        self.fill_in_u_and_v(True, i)
            else:
                return

    def calculate_the_differences(self):
        self.v = np.full(self.n_days, np.nan)
        self.v[0] = 0
        self.u = np.full(self.n_famalies + 1, np.nan)
        self.fill_in_u_and_v(False, 0)
        while any(np.isnan(self.v)) or any(np.isnan(self.u)):
            v_index = np.where(np.isnan(self.v))[0].astype(int)[0]
            self.v[v_index] = 0
            self.fill_in_u_and_v(False, v_index)
        for i in range(self.n_famalies):
            for d in range(self.n_days):
                self.tr_matr[2, d, i] = self.tr_matr[1, d, i] - (self.v[d] + self.u[i])

    def argmin(self, matr):
        min_els = np.ndarray.argmin(matr, axis=1)
        min_el = np.ndarray.argmin(matr[range(self.n_days), min_els])
        # print('(' + str(min_el) + '; ' + str(min_els[min_el]) + '):' + str(tr_matr[2, min_el, min_els[min_el]]))
        return [min_el, min_els[min_el]]

    def pop_2_from_circle(self, circle, amount_to_change, it_is_lead_to_top_border):
        a = circle.pop()
        if a[1] in self.dict_d_as_key[a[0]]:
            added = False
        else:
            added = True
        self.tr_matr[0, a[0], a[1]] += amount_to_change
        if a[1] == self.n_famalies:
            if it_is_lead_to_top_border:
                if self.tr_matr[0, a[0], self.n_famalies] >= 0:
                    self.tr_matr[1, a[0], 0:self.n_famalies] -= 10000
            else:
                sum = np.sum(self.tr_matr[0, a[0], 0:self.n_famalies])
                if sum < 125:
                    self.tr_matr[0, a[0], self.n_famalies] = 125 - sum
                    self.tr_matr[1, a[0], self.n_famalies] = 100000
        self.dict_d_as_key[a[0]].add(a[1])
        self.dict_f_as_key[a[1]].add(a[0])
        a = circle.pop()
        self.tr_matr[0, a[0], a[1]] -= amount_to_change
        if added:
            self.dict_d_as_key[a[0]].remove(a[1])
            self.dict_f_as_key[a[1]].remove(a[0])
        if a[1] == self.n_famalies and not it_is_lead_to_top_border:
            sum = np.sum(self.tr_matr[0, a[0], 0:self.n_famalies])
            if sum >= 125:
                self.tr_matr[1, a[0], self.n_famalies] = 100
        return a[0]

    def find_var_to_except_from_basis(self, d, n):
        min = 1000000
        min_i = -1
        for i in self.dict_d_as_key[d]:
            if i != self.n_famalies:
                if self.tr_matr[0, d, i] == n:
                    cur_min = np.min(self.tr_matr[2, :, i])
                    if cur_min < min:
                        min = cur_min
                        min_i = i
        return min_i

    def lead_to_top_border(self):
        self.set_cost_value()
        for d in range(self.n_days):
            if(d==29):
                print("29")
            lim=self.occupancy_limit_function(d)
            self.tr_matr[0, d, self.n_famalies] = lim - np.sum(self.tr_matr[0, d, 0:self.n_famalies])
            if self.tr_matr[0, d, self.n_famalies] < 0:
                self.tr_matr[1, d, 0:self.n_famalies] += 10000
        next_min = False
        while any(self.tr_matr[0, :, self.n_famalies] < 0):
            if not next_min:
                self.calculate_the_differences()
            next_min = False
            ind_min = self.argmin(self.tr_matr[2, :, 0:self.n_famalies])

            circle = []
            ind_max = np.argmax(self.tr_matr[0, :, ind_min[1]])
            n = self.tr_matr[0, ind_max, ind_min[1]]
            if ind_max in self.days_125 :#and
                if(abs(self.tr_matr[0, ind_max, self.n_famalies])<=8 or abs(self.tr_matr[0, ind_max, self.n_famalies]) != n):
                    max_gift =3000
                    for i in self.dict_d_as_key[ind_max]:
                        if self.tr_matr[0, ind_max, i] == self.tr_matr[0, ind_max, self.n_famalies]:
                            if max_gift < self.tr_matr[1, ind_max, i]:
                                max_gift = self.tr_matr[1, ind_max, i]
                                to_remove = i
                    next_min = True
                    self.tr_matr[2, ind_min[0], ind_min[1]] = 0
                    continue
            circle.append([ind_max, ind_min[1]])  # ---
            circle.append([ind_min[0], ind_min[1]])  # +++
            d_plus = ind_min[0]
            days_already_in_circle = set()
            families_already_in_circle = set()
            days_already_in_circle.add(d_plus)
            while self.tr_matr[0, d_plus, self.n_famalies] < n:
                d_minus = d_plus
                f = self.find_var_to_except_from_basis(d_minus, n)
                if f == -1 or f in families_already_in_circle:  # no famaly in that day
                    if len(circle) == 2 or f in families_already_in_circle:
                        next_min = True
                        break
                    a = circle.pop()  # go back to the previous family and change day by another choice
                    self.tr_matr[2, a[0], a[1]] = 10000
                    f = a[1]

                else:
                    circle.append([d_minus, f])  # ---
                    families_already_in_circle.add(f)
                d_plus = np.argmin(self.tr_matr[2, :, f])
                i = 0
                do_append = True
                while d_plus in days_already_in_circle:  # если
                    self.tr_matr[2, d_plus, f] = 10000
                    d_plus = np.argmin(self.tr_matr[2, :, f])
                    if i > 10:
                        # print('days_already_in_circle')
                        a = circle.pop()
                        self.tr_matr[2, a[0], a[1]] = 10000
                        d_plus = a[0]
                        do_append = False
                        break
                    i += 1
                if do_append:
                    circle.append([d_plus, f])  # +++
                    days_already_in_circle.add(d_plus)
            if next_min:
                self.tr_matr[2, ind_min[0], ind_min[1]] = 0
                continue
            circle.append([d_plus, self.n_famalies])  # ---
            circle.append([ind_max, self.n_famalies])  # +++
            next_min = False
            # print(circle)
            while len(circle) > 0:
                self.pop_2_from_circle(circle, n, True)

    def lead_to_low_border(self):
        self.set_cost_value()
        families_looked_throug = set()
        for d in range(self.n_days):
            sum = np.sum(self.tr_matr[0, d, 0:self.n_famalies])
            if sum < 125:
                # print(' in day ' + str(d) + ' ' + str(np.sum(self.tr_matr[0, d, 0:self.n_famalies])) + ' people')
                self.tr_matr[0, d, self.n_famalies] = 125 - sum
                self.tr_matr[1, d, self.n_famalies] = 100000
        while any(self.tr_matr[1, :, self.n_famalies] == 100000):
            next_min = False
            self.calculate_the_differences()
            ind_min = self.argmin(self.tr_matr[2, :, :])
            ind_max = np.argmax(self.tr_matr[0, :, ind_min[1]])
            n = self.tr_matr[0, ind_max, ind_min[1]]
            circle = []
            if ind_min[1] in families_looked_throug or ind_min[0] in self.days_125:  # and
                while self.tr_matr[ 0, ind_min[0], self.n_famalies] >0 and (self.tr_matr[ 0, ind_min[0], self.n_famalies] < n  or self.tr_matr[ 0, ind_min[0], self.n_famalies] == n + 1):
                    print(f"n = {n}  self.tr_matr[ 0, ind_min[0], self.n_famalies] = {self.tr_matr[ 0, ind_min[0], self.n_famalies]}")
                    self.tr_matr[2, ind_min[0], ind_min[1]] = 0
                    ind_min = self.argmin(self.tr_matr[2, :, :])
                    ind_max = np.argmax(self.tr_matr[0, :, ind_min[1]])
                    n = self.tr_matr[0, ind_max, ind_min[1]]
            circle.append([ind_min[0], self.n_famalies])  # ---
            circle.append([ind_max, self.n_famalies])  # +++
            circle.append([ind_max, ind_min[1]])  # ---
            circle.append([ind_min[0], ind_min[1]])  # +++
            while len(circle) > 0:
                self.pop_2_from_circle(circle, n, False)

    def count(self):
        days_cols = ['day', 'current_choice', 'current_n', 'tax', 'n', 'lim_func', 'free']
        days_data = np.zeros(shape=(100, len(days_cols)))
        days_data[0:100, 0] = list(i for i in range(100))
        self.days = pd.DataFrame(days_data, columns=days_cols)
        self.gifts = 0
        for i in range(self.n_famalies):
            gift = -1
            n = self.data.loc[i, 'n_people']
            d = self.data.loc[i, 'assigned_day'] - 1
            self.days.loc[d, 'n'] += n
            for j in range(self.n_of_choices):
                if d == self.data.loc[i, 'choice_' + str(j)]:
                    self.data.loc[i, 'choice_num'] = j + 1
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
            self.gifts = self.gifts + gift
            # Calculate the gift for not getting top preference
        self.tax = 0
        Ndprev = self.days.loc[99, 'n']
        for d in reversed(range(100)):
            Nd = self.days.loc[d, 'n']
            day_tax = (Nd - 125) / 400 * Nd ** (0.5 + abs(Nd - Ndprev) / 50)
            if Nd < 125 or Nd > 300: day_tax = 30000000
            self.days.loc[d, 'tax'] = day_tax
            lim = self.occupancy_limit_function(d)
            self.days.loc[d, 'lim_func'] = lim
            self.days.loc[d, 'free'] = lim - Nd
            self.tax = self.tax + day_tax
            Ndprev = Nd
        print("tax = ", self.tax)
        print("gifts = ", self.gifts)
        self.total_outgo = self.gifts + self.tax
        print("total outgo = ", self.total_outgo)
        # days.to_csv(f'days_.csv')

    def __repr__(self):
        return ('path  from generation ' + str(self.generation) + ' with total_outgo ' + str(self.total_outgo))

    def save(self, i):
        self.data.to_csv('fam_data_' + str(i) + '.csv', index=False)
        self.days.to_csv(f'days_' + str(i) + '.csv', index=False)


def set_lim(attendance):
    global lim
    days125 = set((62, 63, 64, 68, 69, 70, 71, 75, 76, 77, 78, 82, 83, 84, 85, 89, 90, 91, 92, 96, 97, 98, 99))
    for i in range(len(lim)):
        if i not in days125:
            lim[i] += attendance[i]
            # if lim[i] < attendance[i]:
            # lim[i] = attendance[i]
        else:
            lim[i] += 124


n_population = 4
# lim  = list( min(round(300 - (300-123)/100 * i), 300) for i in range(100))
lim = list(0 for i in range(100))
generation = []
threads = []
start_time = time.perf_counter()
g = 0
fpath = 'fam_data_359549.13406707154.csv'
data = pd.read_csv(fpath)
ind = Individual(data, 'm')
ind.lead_to_borders()
exit()
generation.append(ind)
fpath = 'fam_data_0_liner_upto_309.csv'
# fpath = '_fam_data_0.csv'
data = pd.read_csv(fpath)
ind = Individual(data, 'n')
exit()
generation.append(ind)
fpath = 'fam_data_0_sin.csv'
#fpath = 'fam_data_discrete.csv'
data = pd.read_csv(fpath)
ind = Individual(data, '0')
generation.append(ind)
'''
fpath = '_fam_data_122812.csv'
data = pd.read_csv(fpath)
ind = Individual(data, '1228')
generation.append(ind)
fpath = '_fam_data_123896.csv'
data = pd.read_csv(fpath)
ind = Individual(data, '1238')
generation.append(ind)
#path  from generation 0 with total_outgo 146141.246105369
#path  from generation 1 with total_outgo 171987.13238637708
#path  from generation 2 with total_outgo 173641.13291263825
#path  from generation 3 with total_outgo 173964.22546100037

for i in range(n_population):
    fpath = 'fam_data_'+str(i)+'.csv'
    data = pd.read_csv(fpath)
    #data.assigned_day = data.assigned_day - 1
    ind = Individual(data, i)
    generation.append(ind)
'''
print('data was readed ')

fpath = '_fam_data_123341.csv'
fpath = '_fam_data_123896.csv'
fpath = 'gen1_0.csv'
fpath = 'family_data.csv'
data = pd.read_csv(fpath)

for n in range(10): data['choice_' + str(n)] = data['choice_' + str(n)] - 1


def crossing_over_operator(ind1, ind2):
    global g
    global next_generation
    p1 = Individual(data, g + 1)
    p2 = Individual(data, g + 1)
    for f in range(ind1.n_famalies):
        if (f % 2) == 0:
            p1.set_assigned_day(f, ind1.data.loc[f, 'assigned_day'])
            p2.set_assigned_day(f, ind2.data.loc[f, 'assigned_day'])
        else:
            p2.set_assigned_day(f, ind1.data.loc[f, 'assigned_day'])
            p1.set_assigned_day(f, ind2.data.loc[f, 'assigned_day'])
    next_generation.append(p1)
    next_generation.append(p2)


'''
for _ in range(n_population):
    ind = Individual(data, g)
    t = threading.Thread( target=ind.generate() )
    t.start()
    generation.append(ind)
    threads.append(t)

for t in threads:
    t.join()
threads.clear()
'''
previous_total_outgo = 30000000
while True:
    for ind in generation:
        t = threading.Thread(target=ind.lead_to_borders())
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    threads.clear()

    generation = sorted(generation, key=attrgetter('total_outgo'))
    for ind in generation:
        print(ind)
    current_time = time.perf_counter()
    print(f'time {round((current_time - start_time) / 60, 2)} minutes')

    if previous_total_outgo < generation[0].total_outgo and g > 0:
        break

    i = 0
    for ind in generation:
        ind.save(i)
        i += 1
    next_generation = []
    best_ind = generation.pop(0)
    print(best_ind)
    previous_total_outgo = best_ind.total_outgo
    previous_solution = best_ind.data['assigned_day']

    if g == 1:
      #  lim = [math.ceil((x + 2) / 3) for x in lim]
        print(lim)

    i = 0
    while len(next_generation) < n_population:
        crossing_over_operator(best_ind, generation[i])
        i += 1

    g += 1
    generation.clear()
    generation = next_generation
    del next_generation

'''
submission_cols = ['family_id', 'assigned_day']
submission_data = np.zeros(shape=(len(previous_solution), len(submission_cols)))
submission_data[0:len(previous_solution), 0] = list(i for i in range(len(previous_solution)))
submission = pd.DataFrame(submission_data, columns=submission_cols)
submission['assigned_day'] = previous_solution + 1
submission.to_csv('submission.csv', index=False)
'''