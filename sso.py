import numpy as np
import matplotlib.pyplot as plt
import seaborn

import swarm_intelligence.toolbox as tb

class SharkSearchOptimization():

    ############################################ BUILT-IN ##############################################
    # Class Initiator
    def __init__(self, function, fderiv, ndim, min_value, max_value, size=100, nk=0.7, a=0.7, b=0.7,
                 m=5, task="minimize", iteration=20):
        self._function = function
        self._fderiv = fderiv
        self._ndim = ndim
        self._min_value = min_value
        self._max_value = max_value
        self._size = size
        self._nk = nk
        self._a = a
        self._b = b
        self._m = m
        self._task = "minimize"
        self._iteration = iteration

        self._init_population()

    ############################################# PRIVATE ##############################################
    # Inisiasi Populasi Hiu
    def _init_population(self):
        pop = tb.interval_random(self._size, self._ndim, self._min_value, self._max_value)
        interval = self._max_value - self._min_value
        vel = tb.interval_random(self._size, self._ndim, -(interval/10), (interval/10))

        self._sharks = np.array(pop, np.float32)
        self._velocity = np.array(vel, np.float32)

    # Posisi berdasarkan Gerak Lurus
    def _forward_move(self, current_shark):
        r1 = np.random.random(self._ndim)
        r2 = np.random.random(self._ndim)
        dval = np.zeros(ndim, np.float32)
        vmin = np.vectorize(min)

        for i in range(self._ndim):
            dval[i] = fderiv[i](self._sharks[current_shark])
        if self._task == "minimize":
            dval *= -1
        forward = (self._nk*r1*dval) + (self._a*r2*self._velocity[current_shark])
        max_mov = self._b*self._velocity[current_shark]
        fmovement = vmin(forward, max_mov)
        return fmovement

    # Posisi berdasarkan Gerak Melingkar
    def _rotation_move(self, fmovement):
        rmovement = np.zeros((self._m, self._ndim), np.float32)

        for i in range(self._m):
            rfac = tb.interval_random(1, self._ndim, -1.0, 1.0)
            rmovement[i] = fmovement + rfac*fmovement
        return rmovement

    # Memperbaharui posisi Hiu
    def _update_position(self, current_shark, fmove, rmove):
        choice = np.vstack((fmove, rmove))

        obj = tb.calculate_objective(choice, self._function)
        fit = tb.calculate_fitness(obj, self._task)
        best_choice = tb.find_best(fit)
        self._sharks[current_shark] = choice[best_choice]

    ############################################# PUBLIC ###############################################
    # Fungsi Eksekusi
    def execute(self, show=False, pause=0.2):
        self._objective = tb.calculate_objective(self._sharks, self._function)
        self._fitness = tb.calculate_fitness(self._objective, self._task)
        self._sbest = tb.find_best(self._fitness)

        if show == True:
            plt.style.use('seaborn')
            pt = self._sharks.transpose()
            p = plt.scatter(pt[0], pt[1])
            plt.xlim(self._min_value, self._max_value)
            plt.ylim(self._min_value, self._max_value)
            plt.pause(pause)

        for i in range(self._iteration):
            for cs in range(self._size):
                fmove = self._forward_move(cs)
                rmove = self._rotation_move(fmove)
                self._update_position(cs, fmove, rmove)
            self._objective = tb.calculate_objective(self._sharks, self._function)
            self._fitness = tb.calculate_fitness(self._objective, self._task)
            self._sbest = tb.find_best(self._fitness)

            if show == True:
                pt = self._sharks.transpose()
                p = plt.scatter(pt[0], pt[1])
                plt.xlim(self._min_value, self._max_value)
                plt.ylim(self._min_value, self._max_value)
                plt.pause(pause)
