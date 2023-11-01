import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import math

def V(wait_time): #інтенсивність виходу вимоги з системи
    return 1/wait_time

def Lambda(count,hours): #Інтенсивність вимог
    return count/hours

def Mu(Tserv): #Інтенсивність обслуговування
    return  1/Tserv

def Mu_gen(c,mu,k,v): #Інтенсивність вихідного потоку
    return c*mu + k*v

def Ro(lambdaa, mu_gen): #Завантаженість системи
    return lambdaa/mu_gen

def K_busy(c,q0): #К-ть зайнятих каналів
    sum1 = 0
    for i in range(1,c+1):
        qi = ro**i/math.factorial(i)*q0
        sum1 += i*qi

    sum2=0
    for i in range(0, c + 1):
        qi = ro**i/math.factorial(i)*q0
        sum2 += qi

    return sum1 + c * (1 - sum2)

def A(lambdaa,v,lq): #Абсолютна пропускна здатність
    return lambdaa-(v*lq)

 
def Lq(lambdaa,mu,v,k_busy): #Середня довжина черги
    return ((lambdaa-mu)/v)*k_busy

def Ls(lq, ro): #Середня к-ть вимог 
    return lq + ro

def calculate_product(c, mu, v, k):
    product = 1
    for j in range(1, k + 1):
        product *= c * mu + j * v
    return product

def Q0(ro,c,mu,lambdaa,v): #ймовірність простою
    tolerance=1e-10
    max_iterations=1000
    sum1 = 0
    for i in range(c+1):
        sum1 += (ro ** i) / math.factorial(i)

    sum2=0
    for k in range(1, max_iterations + 1):
        product = calculate_product(c, mu, v, k)
        term = lambdaa ** k / product
        sum2 += term
        if abs(term) < tolerance:
            break

    return (sum1 + (ro**c/math.factorial(c))*sum2)**-1

def Qc(c,ro,q0):
    return ro**c/math.factorial(c)*q0



wait_time = 72 # Обмежений середній час очікування у черзі (години)
speed_per_hour = 2  # Швидкість перекладу (сторінки на годину)
cost_per_page = 10  # Вартість перекладу сторінкиі
work_hours_per_day = 8  # Робочий час на день 
work_days_per_week = 5  # Робочі дні на тиждень
average_pages_per_document = 7  # Середня кількість сторінок у документі
simulation_duration_days = 40  # Тривалість моделювання 
probability_of_sickness = 0.05  # Ймовірність поганого самопочуття (тоді перекладач не працює)

days_of_week = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'П’ятниця']
document_intervals = {
    'Понеділок': [5, 12],
    'Вівторок': [6, 13],
    'Середа': [7, 11],
    'Четвер': [5, 12],
    'П’ятниця': [6, 10]
}

idleness_probabilities = {day: [0] * 3 for day in days_of_week}
unfinished_week = [0,0,0]
rows = 3
cols = 5
leftovers = [[0 for _ in range(cols)] for _ in range(rows)]
time_waiting = [[0 for _ in range(cols)] for _ in range(rows)]
profitsa = [[0 for _ in range(cols)] for _ in range(rows)]
dayn = 0
print("Аналітична частина:\n")
for day in days_of_week:
    document_interval = document_intervals[day]
    a, b = document_interval[0], document_interval[1]
    avg_documents_per_day = (a + b) / 2
    avg_pages_per_day = avg_documents_per_day * average_pages_per_document

    v = V(wait_time)
    lambdaa = Lambda(avg_documents_per_day,work_hours_per_day)

    print(f"День тижня: {day}\n")
    print(f"Інтенсивність вимог: {lambdaa}\n")
    for c in range(1, 4):
        
        docspeed = average_pages_per_document/speed_per_hour
        mu = Mu(docspeed)
        mu_gen = Mu_gen(c,mu,avg_documents_per_day,v)
        ro = Ro(lambdaa,mu_gen)
        q0 = Q0(ro,c,mu,lambdaa,v)
        idleness_probabilities[day][c - 1] = q0
        qc = Qc(c,ro,q0)
        k_busy = K_busy(c,q0)
        lq = Lq(lambdaa,mu*c,v,k_busy)
        lqc = Lq(lambdaa,mu,v,k_busy)
        a = A(lambdaa,v,lq)
        ab = A(lambdaa,v,lqc)
        ls = Ls(lq, ro)
        profit = abs(cost_per_page*avg_pages_per_day * ab)
        profitsa[c-1][dayn] = profit
        ability = work_hours_per_day * c / docspeed

        if (dayn != 0):
            for d in range(0,dayn):
                if ((ability - leftovers[c-1][d]) >= 0):
                    ability = ability - leftovers[c-1][d]
                    leftovers[c-1][d] = 0
                else:
                    leftovers[c-1][d] = leftovers[c-1][d] - ability
                    ability = 0

        if (dayn != 0):
            for t in range(0,dayn):
                time_waiting[c-1][t]+=1
                if (time_waiting[c-1][t]==3):
                    unfinished_week[c-1]+=leftovers[c-1][t]
                    leftovers[c-1][t] = 0

        unfinished_today = avg_documents_per_day - ability
        leftovers[c-1][dayn] = unfinished_today       
        if (dayn == 4):
            for l in range(0,dayn+1):
                unfinished_week[c-1] += leftovers[c-1][l]
        
        print(f"К-ть перекладачів: {c}  Інтенсивність обслуговування: {mu*c}\n")
        print(f"К-ть перекладачів: {c}  Абсолютна пропускна здатність: {a}\n")
        print(f"К-ть перекладачів: {c}  Кількість зайнятих каналів: {k_busy}\n")
        print(f"К-ть перекладачів: {c}  Середня довжина черги : {lq}\n")
        print(f"К-ть перекладачів: {c}  Середня кількість вимог: {ls}\n")
        print(f"К-ть перекладачів: {c}  Ймовірність простою системи: {q0}\n")

    dayn +=1
    print(f"----------------------------------------------")

for c in range(1,4):
    print(f"Втрата документів за тиждень для {c} перекладачів = {unfinished_week[c-1]}")


c_values = range(1, 4)
width = 0.2
x = np.arange(len(days_of_week))
fig, ax = plt.subplots()
for i, c in enumerate(c_values):
    q0_values = [idleness_probabilities[day][i] for day in days_of_week]
    ax.bar(x + i * width, q0_values, width, label=f"{c} ")
ax.set_xlabel("День тижня")
ax.set_ylabel("Ймовірність простою (q0)")
ax.set_title("Ймовірність простою для кожної кількості перекладачів")
ax.set_xticks(x + 0.2)
ax.set_xticklabels(days_of_week)
ax.legend(title="Кількість перекладачів (c)")
plt.show()

print("Ітераційна частина:\n")
current_day = 0
rows = 3
cols = 5
leftovers = [[0 for _ in range(cols)] for _ in range(rows)]
time_waiting = [[0 for _ in range(cols)] for _ in range(rows)]
dayn = 0
unfinished_weekb = [0,0,0]
profits = [[0 for _ in range(cols)] for _ in range(rows)]
while current_day < simulation_duration_days:
    day_index = current_day % len(days_of_week)
    day_of_week = days_of_week[day_index]
    document_count = np.random.randint(document_intervals[day_of_week][0], document_intervals[day_of_week][1] + 1)
    pages_to_translate = np.random.poisson(average_pages_per_document) * document_count
    for c in range(1, 4):
        docspeed = np.random.poisson(average_pages_per_document)/speed_per_hour
        ability = work_hours_per_day * c / docspeed
        profit = abs(cost_per_page*ability*np.random.poisson(average_pages_per_document))
        profits[c-1][dayn] += profit
        if (dayn != 0):
            for d in range(0,dayn):
                if ((ability - leftovers[c-1][d]) >= 0):
                    ability = ability - leftovers[c-1][d]
                    leftovers[c-1][d] = 0
                else:
                    leftovers[c-1][d] = leftovers[c-1][d] - ability
                    ability = 0

        if (dayn != 0):
            for t in range(0,dayn):
                time_waiting[c-1][t]+=1
                if (time_waiting[c-1][t]==3):
                    unfinished_weekb[c-1]+=leftovers[c-1][t]
                    leftovers[c-1][t] = 0

        unfinished_today = avg_documents_per_day - ability
        leftovers[c-1][dayn] = unfinished_today       
        if (dayn == 4):
            for l in range(0,dayn+1):
                unfinished_weekb[c-1] += leftovers[c-1][l]
    dayn +=1
    if (dayn==5):
        dayn=0
    current_day+=1

for i in range(len(unfinished_week)):
    unfinished_weekb[i] = unfinished_weekb[i] / 8

for c in range(1,4):
    unfinished_weekb[c-1] = abs(unfinished_weekb[c-1])
    print(f"Втрата документів за тиждень для {c} перекладачів = {unfinished_weekb[c-1]}")

result_earn = [[element / 8 for element in row] for row in profits]


num_translators = len(profitsa)
num_days = len(profitsa[0])
width = 0.2
x = np.arange(num_days)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for i in range(num_translators):
    ax1.bar(x + i * width, profitsa[i], width, label=f'{i + 1}')

ax1.set_xlabel('День тижня')
ax1.set_ylabel('Прибуток')
ax1.set_title('Прибуток для кожної к-ті перекладачів (аналітично)')
ax1.set_xticks(x + width * (num_translators - 1) / 2)
ax1.set_xticklabels(days_of_week)
ax1.legend(title='К-ть перекладачів')

for i in range(num_translators):
    ax2.bar(x + i * width, result_earn[i], width, label=f'{i + 1} ')
ax2.set_xlabel('День тижня')
ax2.set_ylabel('Прибуток')
ax2.set_title('Прибуток для кожної к-ті перекладачів (ітераційно)')
ax2.set_xticks(x + width * (num_translators - 1) / 2)
ax2.set_xticklabels(days_of_week)
ax2.legend(title='К-ть перекладачів')
fig.suptitle('Порівняння аналітичного і ітераційного методів')
plt.show()



translators = range(1, len(unfinished_week) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(translators, unfinished_week)
ax1.set_xlabel('К-ть перекладачів')
ax1.set_ylabel('Втрата документів за тиждень')
ax1.set_title('Аналітичний метод')
ax1.set_xticks(translators)


ax2.bar(translators, unfinished_weekb)
ax2.set_xlabel('К-ть перекладачів')
ax2.set_ylabel('Втрата документів за тиждень')
ax2.set_title('Ітераційний метод')
ax2.set_xticks(translators)
fig.suptitle('Порівняння аналітичного і ітераційного методів')
plt.show()
