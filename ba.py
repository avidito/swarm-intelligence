import numpy as np
import matplotlib.pyplot as plt
import seaborn

import swarm_intelligence.toolbox as tb

class BatAlgorithm():

    ############################################ BUILT-IN ##############################################
    # Class Initiator
    def __init__(self, function, ndim, min_value, max_value, size=100, task="minimize", iteration=20,
                 f_interval=[0,1], a_interval=[0,1], r_interval=[0,1], alpha=0.8, gamma=0.8):
        self.function = function
        self.ndim = ndim
        self.min_value = min_value
        self.max_value = max_value
        self.size = size
        self.task = task
        self.iteration = iteration
        self.f_interval = f_interval
        self.a_interval = a_interval
        self.r_interval = r_interval
        self.alpha = alpha
        self.gamma = gamma

        self._init_population()
        half_aint = a_interval[0] + ((a_interval[1] - a_interval[0])/2)
        half_rint = r_interval[0] + ((r_interval[1] - r_interval[0])/2)
        self._a = tb.interval_random(1, size, half_aint, a_interval[1])[0]
        self._r = tb.interval_random(1, size, r_interval[0], half_rint)[0]

    ############################################# PRIVATE ##############################################
    # Inisiasi Populasi
    def _init_population(self):
        interval = self.max_value - self.min_value

        self._bats = tb.interval_random(self.size, self.ndim, self.min_value, self.max_value)
        self._velocity = tb.interval_random(self.size, self.ndim, self.min_value, self.max_value)

    # Memperbaharui Kecepatan Kelelawar
    def _update_velocity(self, bat):
        beta = np.random.random(self.ndim)

        fbat = self.f_interval[0] + (self.f_interval[1] - self.f_interval[0])*beta
        self._velocity[bat] = 0.5*self._velocity[bat] + (self._bats[self.bbest] - self._bats[bat])*fbat

    # Aproksimasi Posisi Pencarian Lokal
    def _local_search(self, new_pos, a_average):
        rand = np.random.random()
        return new_pos + rand*a_average

    ############################################# PUBLIC ###############################################
    # Fungsi Eksekusi
    def execute(self, show=False, pause=0.2):
        self._objective = tb.calculate_objective(self._bats, self.function)
        self._fitness = tb.calculate_fitness(self._objective, self.task)
        self.bbest = tb.find_best(self._fitness)

        if show == True:
            plt.style.use('seaborn')
            pt = self._bats.transpose()
            p = plt.scatter(pt[0], pt[1])
            plt.xlim(self.min_value, self.max_value)
            plt.ylim(self.min_value, self.max_value)
            plt.pause(pause)

        for i in range(self.iteration):
            for bat in range(self.size):
                self._update_velocity(bat)
                new_pos = self._bats[bat] + self._velocity[bat]

                r1 = np.random.random()
                if r1 > self._r[bat]:
                    a_average = np.average(self._a)
                    new_pos = self._local_search(new_pos, a_average)
                vvc = np.vectorize(tb.value_clip)
                new_pos = vvc(new_pos, self.min_value, self.max_value)
                new_obj = tb.calculate_objective(new_pos.reshape(1,-1), self.function)
                new_fit = tb.calculate_fitness(new_obj, self.task)

                r2 = np.random.random()
                if new_fit > self._fitness[bat] and r2 < self._a[i]:
                    self._bats[bat] = new_pos
                    self._objective[bat] = new_obj
                    self._fitness[bat] = new_fit
                    self._a[bat] *= self.alpha
                    self._r[bat] = self.r_interval[1]*(1 - (1/(1 + self.gamma*i)))
            self._bbest = tb.find_best(self._fitness)

            if show == True:
                pt = self._bats.transpose()
                p = plt.scatter(pt[0], pt[1])
                plt.xlim(self.min_value, self.max_value)
                plt.ylim(self.min_value, self.max_value)
                plt.pause(pause)
