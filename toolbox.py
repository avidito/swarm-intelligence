import numpy as np

# Angka Acak dengan interval
def interval_random(row, col, min_val, max_val):
    ran = abs(max_val - min_val)
    return ran * np.random.random((row, col)) + min_val

# Perhitungan Nilai Objektif
def calculate_objective(data, function, round_point=10):
        row, col = data.shape
        objective = np.zeros((row), np.float32)

        for i in range(row):
            objective[i] = round(function(data[i]), round_point)
        return objective

# Perhitungan Nilai Fitness
def calculate_fitness(objective, task, target=None):
    size = objective.shape
    fitness = np.copy(objective)

    if target != None:
        error = 0.0
        for i in range(size):
            fitness[i] = abs(target - objective[i])
            error += fitness[i]**2
        error = sqrt(error)
    vround = np.vectorize(round)
    fitness = vround(task(fitness), 10)

    if target == None:
        return fitness
    else:
        return fitness, error

# Interpretasi Maximize
def maximize(val):
    return abs(val)

# Interpretasi Minimize
def minimize(val):
    return abs(1/(1+val))

# Pemotongan Nilai
def value_clip(val, min_value, max_value):
    if val < min_value:
        return min_value
    if val > max_value:
        return max_value
    return val

# Roulete Wheel Selection
def roullete_wheel(prob):
    total = 0.0
    for i in range(len(prob)):
        total += prob[i]

    prob = prob/total
    choice = []
    c = 0

    for i in range(len(prob)):
        c += prob[i]
        choice.append(c)
    r = np.random.random()
    for i in range(len(choice)):
        if r < choice[i]:
            return i

# Edge Adjacent
def get_adjacent(s, dis, ex=None):
    sz = len(dis[s])
    adj = []

    for j in range(sz):
        if j == s or j == ex:
            continue
        if j < s:
            o = j
            d = s
        else:
            o = s
            d = j
        if dis[o,d] != 0:
            adj.append([o,d])
    return adj
